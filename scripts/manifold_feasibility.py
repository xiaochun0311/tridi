"""Seven paper feasibility pilots on the existing TriDi checkpoint; no retraining."""
import argparse
import hashlib
import json
import pickle
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from scipy.cluster.vq import kmeans2
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from tridi.data.batch_data import BatchData
from tridi.model import get_model
from tridi.model.conditioning.contact_ae_clip import ContactEnc, ContactDec
from tridi.model.wrappers.mesh import MeshModel
from tridi.utils.geometry import matrix_to_rotation_6d, rotation_6d_to_matrix

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cuda")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--data-root", default="data/preprocessed/grab_smplh")
parser.add_argument("--test-fps", type=int, default=1)
parser.add_argument("--preview-dir", default=None)
args = parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device(args.device)
root = Path(args.data_root)
report_path = Path("docs/manifold_feasibility_results.json")
report = {"completed":False, "data_root":str(root), "train_fps":10, "test_fps":args.test_fps, "scope": "banana pilot; train s1, held-out s9, multiple actions; adjacent frames correlated; no retraining",
          "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
          "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "gitignore_sha256": hashlib.sha256(Path(".gitignore").read_bytes()).hexdigest(),
          "checkpoint": "assets/gb_main.pth", "seeds": [42, 43, 44], "steps": args.steps,
          "score_caveat": "VP-to-VE Tweedie adapter is approximate for TriDi's L1 subject/object predictor.",
          "results": {}}
assert report["branch"] == "manifold-test", report["branch"]
report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False))

def save(key, value):
    report["results"][key] = value
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(key, json.dumps(value, ensure_ascii=False), flush=True)

cfg = OmegaConf.load("experiments/001_gb_main/config.yaml")
model = get_model(cfg).to(device).eval()
model.load_state_dict(torch.load("assets/gb_main.pth", map_location=device, weights_only=False)["model"], strict=True)
weights = torch.load("assets/gb_contacts.pth", map_location=device, weights_only=False)
enc, dec = ContactEnc(690, 256, 128).to(device).eval(), ContactDec(690, 256, 128).to(device).eval()
enc.load_state_dict(weights["model_enc"], strict=True)
dec.load_state_dict(weights["model_dec"], strict=True)
idx = np.load("assets/smpl_template_decimated_idxs.npy")
kpts = np.load(root / "object_keypoints/banana.npz")["cartesian"]
with open("data/preprocessed/grab_smplh/object_pointnext.pkl", "rb") as f:
    pointnext = torch.as_tensor(pickle.load(f)["banana"], dtype=torch.float32, device=device).reshape(1, -1)

def load(split):
    rows, frames, sequences = [], [], []
    fps = 10 if split == "train" else args.test_fps
    with h5py.File(root / f"dataset_{split}_{fps}fps.hdf5") as f:
        for subject in ["s1" if split == "train" else "s9"]:
            for action in sorted(f[subject]):
                if not action.startswith("banana_"):
                    continue
                seq = f[subject][action]
                sequences.append({"subject":subject,"sequence":action,"frames":int(seq.attrs["T"]),
                                  "prep_s_range":[float(seq["prep_s"][:].min()),float(seq["prep_s"][:].max())]})
                for i in range(int(seq.attrs["T"])):
                    rotations = np.concatenate([seq["sbj_smpl_global"][i].reshape(1,3,3),
                        seq["sbj_smpl_body"][i].reshape(21,3,3), seq["sbj_smpl_lh"][i].reshape(15,3,3),
                        seq["sbj_smpl_rh"][i].reshape(15,3,3)], axis=0)
                    pose = matrix_to_rotation_6d(rotations).numpy().reshape(-1)
                    obj_r = seq["obj_R"][i].reshape(3,3)
                    obj_c = seq["obj_c"][i]
                    posed = kpts @ obj_r.T + obj_c
                    distance = cKDTree(posed).query(seq["sbj_v"][i][idx], workers=1)[0]
                    with torch.no_grad():
                        contact = enc(torch.as_tensor(distance < .05, dtype=torch.float32, device=device)[None]).cpu().numpy()[0]
                    rows.append(np.concatenate([seq["sbj_smpl_betas"][i], pose, seq["sbj_smpl_transl"][i],
                        matrix_to_rotation_6d(obj_r).numpy(), obj_c, contact]))
                    frames.append(f"{subject}/{action}/{i}")
    report.setdefault("sequences", {})[split] = sequences
    return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=device), frames

train, train_frames = load("train")
test, test_frames = load("test")
chosen = np.linspace(0, len(test)-1, 4, dtype=int).tolist()
x0 = test[chosen]
report.update(train_frames=len(train), test_frames=len(test), chosen_frames=[test_frames[i] for i in chosen],
              contact_source="stored preprocessed human vertices; canonical object keypoints; threshold 0.05")
alpha = model.scheduler.alphas_cumprod.to(device)
sigmas = ((1-alpha)/alpha).sqrt()
calls = 0

@torch.no_grad()
def predict_vp(x, t):
    global calls
    calls += 1
    times = torch.full((len(x),), int(t), dtype=torch.long, device=device)
    inp = model.get_input_with_conditioning(x[:, :334], t=times, t_aux=times,
        obj_group=torch.full((len(x),), 20, dtype=torch.long, device=device),
        obj_pointnext=pointnext.expand(len(x), -1), contact_map=x[:, 334:])
    return model.denoising_model(inp, times, times, times)

@torch.no_grad()
def score(z, t):
    return (predict_vp(alpha[t].sqrt()*z, t)-z)/sigmas[t].square()

def rotations(x):
    return torch.cat([x[:, 10:322].reshape(-1,52,6), x[:,325:331].reshape(-1,1,6)], dim=1)

def put_rotations(x, r):
    y = x.clone()
    y[:,10:322], y[:,325:331] = r[:,:52].reshape(-1,312), r[:,52]
    return y

def canonical(x):
    return put_rotations(x, matrix_to_rotation_6d(rotation_6d_to_matrix(rotations(x))))

def skew(w):
    a,b,c = w.unbind(-1)
    zero = torch.zeros_like(a)
    return torch.stack([zero,-c,b,c,zero,-a,-b,a,zero], dim=-1).reshape(*w.shape[:-1],3,3)

def tangent_basis(x):
    r = rotation_6d_to_matrix(rotations(x))
    columns = []
    for j in range(3):
        w = torch.zeros((*r.shape[:-2],3),device=device)
        w[...,j] = .001
        columns.append((matrix_to_rotation_6d(torch.matrix_exp(skew(w))@r)-matrix_to_rotation_6d(torch.matrix_exp(skew(-w))@r))/.002)
    return torch.linalg.qr(torch.stack(columns, dim=-1)).Q

def decompose(v, u):
    packed = rotations(v)
    tang = (u @ (u.transpose(-1,-2) @ packed.unsqueeze(-1))).squeeze(-1)
    return tang, packed-tang

def raw_rotation_error(x):
    return float((rotations(x)-rotations(canonical(x))).square().mean().sqrt())

def angle(r1, r2):
    tr = (r1.transpose(-1,-2)@r2).diagonal(dim1=-2,dim2=-1).sum(-1)
    return torch.acos(((tr-1)/2).clamp(-1,1))

batch = BatchData(sbj=["pilot"],obj_class=torch.tensor([20],device=device),
                 obj_group=torch.tensor([20],device=device),obj_pointnext=pointnext)
mesh = MeshModel("data/smplx_models",1,{0:trimesh.load(root/"object_meshes/banana.ply",process=False)},
                 {0:{"cartesian":kpts}},device)
zero_class = torch.zeros(1,dtype=torch.long,device=device)

def physical(x):
    return mesh.get_meshes_wkpts_th(model.split_output(x),zero_class,scale=None,sbj_gender=torch.ones(1,dtype=torch.bool,device=device))

def contact_cost(x, mask):
    vertices, points = physical(x)
    distances = torch.cdist(vertices[:,idx],points).min(-1).values
    return (distances*mask).sum()/mask.sum().clamp_min(1)

@torch.no_grad()
def sample_metrics(x, fixed_mask=None):
    mask = (dec(x[:,334:])>.4).float()
    evaluated_mask = mask if fixed_mask is None else fixed_mask
    return {"finite":bool(torch.isfinite(x).all()),"raw6d_rms":raw_rotation_error(x),
            "decoded_contact_vertices":int(mask.sum()),"evaluated_contact_vertices":int(evaluated_mask.sum()),
            "fixed_mask_contact_distance":float(contact_cost(x,evaluated_mask)) if evaluated_mask.sum() else None,
            "nearest_train_parameter_rms":float(torch.cdist(x[:,:334],train[:,:334]).min()/np.sqrt(334))}

def sample_ve(seed, steps, mad=False, smooth=False):
    torch.manual_seed(seed)
    z = torch.cat([torch.randn(1,325,device=device),torch.randn(1,9,device=device),
                   torch.randn(1,128,device=device)],dim=1)/alpha[999].sqrt()
    times = np.rint(np.linspace(999,0,steps+1)).astype(int)
    for t, tn in zip(times[:-1],times[1:]):
        sigma = float(sigmas[t]); ds = float(sigmas[tn])-sigma
        if smooth:
            v = torch.randn_like(z)
            v /= v.norm()
            s = .5*(score(z+.03*v,t)+score(z-.03*v,t))
        else:
            s = score(z,t)
        if mad:
            target = sigma*1.1 if t < 999 else sigma*.9
            tp = int((sigmas-target).abs().argmin())
            if tp == t: tp = t+1 if t < 999 else t-1
            actual_delta = float(sigmas[tp])-sigma
            derivative = (score(z,tp)-s)/actual_delta
            # Algorithm 1, p=2 and explicitly untuned a=b=1.
            gamma = sigma*sigma/2
            m = 1/(1+gamma-gamma/(sigma*sigma))
            z -= m*ds*sigma*((1+gamma)*s+gamma/(2*sigma)*derivative)
        else:
            z -= ds*sigma*s
    return predict_vp(alpha[0].sqrt()*z,0)

start = time.time()
baseline_outputs = []
baseline_rows, coarse_rows, official_rows, mad_rows, smooth_rows = [], [], [], [], []
for seed in report["seeds"]:
    calls = 0
    baseline = sample_ve(seed,2*args.steps)
    common_mask = (dec(baseline[:,334:])>.4).float().detach()
    baseline_rows.append({"seed":seed,"nfe":calls,**sample_metrics(baseline,common_mask)})
    calls = 0
    coarse = sample_ve(seed,args.steps)
    coarse_rows.append({"seed":seed,"nfe":calls,**sample_metrics(coarse,common_mask)})
    calls = 0
    modified = sample_ve(seed,args.steps,mad=True)
    mad_rows.append({"seed":seed,"nfe":calls,**sample_metrics(modified,common_mask)})
    calls = 0
    smoothed = sample_ve(seed,args.steps,smooth=True)
    smooth_rows.append({"seed":seed,"nfe":calls,**sample_metrics(smoothed,common_mask)})
    torch.manual_seed(seed)
    official = model.forward_sample("111",batch,scheduler="ddim",num_inference_steps=2*args.steps,disable_tqdm=True)
    baseline_outputs.append(official)
    official_rows.append({"seed":seed,"nfe":2*args.steps,**sample_metrics(official,common_mask)})
    if args.preview_dir and seed == 42:
        destination = Path(args.preview_dir) / "seed42"
        destination.mkdir(parents=True, exist_ok=True)
        arrays = {"subject_faces":mesh.get_faces_np(),
                  "common_contacts":idx[common_mask[0].cpu().numpy().astype(bool)],
                  "object_vertices":mesh.canonical_obj_meshes[0].vertices,
                  "object_faces":mesh.canonical_obj_meshes[0].faces}
        for name, output in [("euler",baseline),("mad",modified),("smooth",smoothed),("ddim",official)]:
            with torch.no_grad():
                subjects, objects = mesh.get_meshes(model.split_output(output),zero_class,scale=None,
                    sbj_gender=torch.ones(1,dtype=torch.bool,device=device))
            subjects[0].export(destination / f"{name}_human.ply")
            objects[0].export(destination / f"{name}_object.ply")
            arrays[f"{name}_human_vertices"] = subjects[0].vertices
            arrays[f"{name}_object_R"] = rotation_6d_to_matrix(output[:,325:331])[0].cpu().numpy()
            arrays[f"{name}_object_c"] = output[0,331:334].cpu().numpy()
        np.savez_compressed(destination / "preview_arrays.npz",**arrays)
        print("MESH_PREVIEW_EXPORTED",destination,flush=True)

save("1_MAD",{"baseline":"VP-to-VE ordinary Euler","adaptation":"Algorithm 1 p=2,a=b=1; fixed-z finite difference at distinct trained sigma",
    "baseline_rows":baseline_rows,"coarse_Euler_resolution_control":coarse_rows,"official_DDIM_reference":official_rows,
    "MAD_rows":mad_rows,"seconds":time.time()-start, "MAD_boundary_derivative":"backward difference at t999; forward distinct-sigma difference elsewhere",
    "evaluation":"same seed, same initial VE state and baseline contact mask for Euler/MAD/smoothing; contact distance in model coordinates",
    "limitation":"Untuned pilot; approximate score because subject/object x0 predictor uses L1; contact metric is not penetration or overall quality."})
save("3_log_domain_smoothing",{"method":"symmetric score average at z +/- 0.03 unit random direction; same total NFE as Euler baseline",
    "baseline_rows":baseline_rows,"coarse_Euler_resolution_control":coarse_rows,"smoothed_rows":smooth_rows,
    "limitation":"This evaluates score smoothing; it does not prove geometry adaptation, robustness or generation quality."})

geo_rows = []
torch.manual_seed(123)
for scale in [.05,.2,.5]:
    r = rotation_6d_to_matrix(rotations(x0))
    w = torch.randn((*r.shape[:-2],3),device=device)*scale
    intrinsic = matrix_to_rotation_6d(torch.matrix_exp(skew(w))@r)
    extrinsic = rotations(x0)+torch.randn_like(rotations(x0))*scale
    geo_rows.append({"noise_scale":scale,"intrinsic_raw6d_rms":float((intrinsic-matrix_to_rotation_6d(rotation_6d_to_matrix(intrinsic))).square().mean().sqrt()),
        "extrinsic_raw6d_rms":float((extrinsic-matrix_to_rotation_6d(rotation_6d_to_matrix(extrinsic))).square().mean().sqrt()),
        "intrinsic_mean_angle_radians":float(angle(r,rotation_6d_to_matrix(intrinsic)).mean()),
        "extrinsic_mean_angle_radians":float(angle(r,rotation_6d_to_matrix(extrinsic)).mean())})
save("2_symmetric_manifolds",{"stage":"SO(3) representation/noising prerequisite only","rows":geo_rows,
    "limitation":"The paper projects OU diffusion and learns Ito drift/covariance. Expmap noising is not that method; retraining needed. TriDi already projects decoded rotations."})

jac_rows = []
torch.manual_seed(321)
q = torch.linalg.qr(torch.randn(462,16,device=device)).Q.T
for t in [25,100,250,500]:
    point = alpha[t].sqrt()*x0[:1]
    mats = []
    for h in [.002,.001]:
        response = (predict_vp(point+h*q,t)-predict_vp(point-h*q,t))/(2*h)
        mats.append(response.T)
    sv = torch.linalg.svdvals(mats[1])
    weights = sv/sv.sum()
    jac_rows.append({"t":t,"direction_count":16,"projected_effective_rank":float(torch.exp(-(weights*weights.clamp_min(1e-12).log()).sum())),
        "projected_log_volume":float(sv.clamp_min(1e-12).log().sum()),
        "finite_difference_relative_change":float((mats[0]-mats[1]).norm()/mats[1].norm()),
        "singular_values":sv.tolist()})
save("4_local_geometry",{"stage":"denoiser Jacobian in 16 orthonormal directions","rows":jac_rows,
    "limitation":"Projected directional spectrum is not full manifold dimension/log volume. No causal quality claim or exact activation-region complexity."})

rate_rows = []
v = torch.zeros_like(x0[:1]); n = torch.zeros_like(v)
ut = tangent_basis(x0[:1])
v[:,10:16] = ut[:,0,:,0]
torch.manual_seed(8)
rand = torch.randn(1,6,device=device)
normal = rand-(ut[:,0]@(ut[:,0].transpose(-1,-2)@rand.unsqueeze(-1))).squeeze(-1)
n[:,10:16] = normal/normal.norm()
for t in [25,100,250,500]:
    h = .001
    dt = (score(x0[:1]+h*v,t)-score(x0[:1]-h*v,t))/(2*h)
    dn = (score(x0[:1]+h*n,t)-score(x0[:1]-h*n,t))/(2*h)
    rate_rows.append({"t":t,"sigma":float(sigmas[t]),"tangent_restoring_derivative":float(-(dt*v).sum()),
                      "normal_restoring_derivative":float(-(dn*n).sum())})
chain_rows = []
t = 100; sigma = float(sigmas[t]); eta = .05*sigma*sigma
for exponent in [0.,1.]:
    torch.manual_seed(42)
    z = x0[:1]+sigma*torch.randn_like(x0[:1])
    initial = raw_rotation_error(z)
    for _ in range(20):
        z = z+eta*sigma**exponent*score(z,t)+np.sqrt(2*eta)*torch.randn_like(z)
    chain_rows.append({"drift_exponent":exponent,"steps":20,"eta":eta,"initial_raw6d_rms":initial,
                       "final_raw6d_rms":raw_rotation_error(z),"finite":bool(torch.isfinite(z).all()),
                       "rms_displacement":float((z-x0[:1]).square().mean().sqrt())})
save("5_rate_separations",{"rates":rate_rows,"fixed_noise_langevin":chain_rows,
    "limitation":"Known SO(3) rotational tangent is only part of the data manifold. L1 predictor is approximate score; short chains cannot prove stationary uniform sampling or asymptotic rates."})

a = train[:,10:334].cpu().numpy().astype(np.float64)
b = test[:,10:334].cpu().numpy().astype(np.float64)
block_scale = np.sqrt(np.mean(np.var(a,axis=0))).clip(.05)
a, b = a/block_scale,b/block_scale
_, _, vh = np.linalg.svd(a-a.mean(0),full_matrices=False)
global_basis = vh[:4].T

def density_fit(data, basis=None):
    mean = data.mean(0); centered = data-mean
    if basis is None:
        _,_,vh = np.linalg.svd(centered,full_matrices=False)
        basis = vh[:4].T
    coord = centered@basis
    residual = max(float((np.square(centered).sum()-np.square(coord).sum())/(len(data)*(data.shape[1]-4))),1e-3)
    variances = np.maximum(np.mean(coord*coord,axis=0),residual)
    return mean,basis,residual,variances

def logpdf(data, fit):
    mean,basis,residual,variances = fit
    delta = data-mean; coord = delta@basis
    residual_norm = np.maximum(np.square(delta).sum(1)-np.square(coord).sum(1),0)
    return -.5*(data.shape[1]*np.log(2*np.pi)+np.log(variances).sum()+(data.shape[1]-4)*np.log(residual)
                +(np.square(coord)/variances).sum(1)+residual_norm/residual)

density_rows = []
for seed in [42,43,44]:
    for k in [1,2,4]:
        labels = np.zeros(len(a),dtype=int) if k == 1 else kmeans2(a,k,minit="++",seed=seed,iter=30)[1]
        for local in [False,True]:
            components = []
            sizes = []
            for j in range(k):
                group = a[labels==j]
                if len(group)<6: break
                components.append(density_fit(group,None if local else global_basis)); sizes.append(len(group))
            if len(components)!=k: continue
            ll = logsumexp(np.stack([logpdf(b,f)+np.log(n/len(a)) for f,n in zip(components,sizes)]),axis=0)
            density_rows.append({"seed":seed,"components":k,"basis":"local" if local else "shared_global",
                "rank_per_component":4,"cluster_frames":sizes,"heldout_mean_log_likelihood":float(ll.mean())})
save("6_multi_subspace",{"stage":"single/shared-subspace/local-subspace Gaussian mixtures fitted only on train frames",
    "rows":density_rows,"limitation":"PPCA mixtures are a prerequisite pilot, not trained diffusion MoE; local bases add parameters. One object/one training subject cannot justify broad object/action experts."})

def yaw(x, theta):
    w = torch.tensor([0.,0.,theta],device=device)
    r = torch.matrix_exp(skew(w))
    y = x.clone()
    y[:,10:16] = matrix_to_rotation_6d(r@rotation_6d_to_matrix(x[:,10:16]))
    y[:,325:331] = matrix_to_rotation_6d(r@rotation_6d_to_matrix(x[:,325:331]))
    y[:,322:325] = x[:,322:325]@r.T
    y[:,331:334] = x[:,331:334]@r.T
    return y

yaw_rows = []
pred = predict_vp(x0,0)
for theta in [np.pi/2,np.pi]:
    actual, expected = predict_vp(yaw(x0,theta),0),yaw(pred,theta)
    yaw_rows.append({"yaw_radians":theta,"root_angle_error_radians":float(angle(rotation_6d_to_matrix(actual[:,10:16]),rotation_6d_to_matrix(expected[:,10:16])).mean()),
        "object_angle_error_radians":float(angle(rotation_6d_to_matrix(actual[:,325:331]),rotation_6d_to_matrix(expected[:,325:331])).mean()),
        "translation_rms_error":float(torch.cat([actual[:,322:325]-expected[:,322:325],actual[:,331:334]-expected[:,331:334]],dim=1).square().mean().sqrt()),
        "contact_latent_rms_error":float((actual[:,334:]-expected[:,334:]).square().mean().sqrt())})

initial = canonical(baseline_outputs[0].detach())
fixed_mask = (dec(initial[:,334:])>.4).float().detach()
guidance_rows = []
for kind in ["euclidean","rotation_tangent"]:
    current = initial.clone()
    costs = [float(contact_cost(current,fixed_mask))]
    angles = []
    for _ in range(3):
        current = current.detach().requires_grad_(True)
        grad, = torch.autograd.grad(contact_cost(current,fixed_mask),current)
        grad[:,:10],grad[:,334:] = 0,0
        if kind == "rotation_tangent":
            tang,_ = decompose(grad,tangent_basis(current.detach()))
            grad = put_rotations(grad,tang)
        grad = grad/grad.norm().clamp_min(1e-12)
        old_r = rotation_6d_to_matrix(rotations(current.detach()))
        candidates = [current.detach()] + [current.detach()-lr*grad for lr in [.01,.03,.1]]
        with torch.no_grad():
            values = [float(contact_cost(c,fixed_mask)) for c in candidates]
            current = canonical(candidates[int(np.argmin(values))])
        angles.append(float(angle(old_r,rotation_6d_to_matrix(rotations(current))).mean()))
        costs.append(float(contact_cost(current,fixed_mask)))
    guidance_rows.append({"kind":kind,"gradient_evaluations":3,"fixed_contact_vertices":int(fixed_mask.sum()),
                           "contact_cost_history":costs,"mean_rotation_step_radians":angles})
save("7_STRiDE",{"stage":"clean-input yaw consistency and embedded SO(3) tangent projection with canonical-6D retraction",
    "yaw_rows":yaw_rows,"guidance_rows":guidance_rows,
    "limitation":"SMPL SO(3) states are not a robot joint torus. No trajectories, joint limits, equivariant training or planning benchmark; static contact improvement is only a transferable component."})
report["completed"] = True
report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
subject, sequence, frame = test_frames[0].split("/")
with h5py.File(root / f"dataset_test_{args.test_fps}fps.hdf5") as f:
    stored = torch.as_tensor(f[subject][sequence]["sbj_v"][int(frame)],device=device)
with torch.no_grad():
    rebuilt, _ = physical(test[:1])
report["coordinate_check"] = {"representative_frame":test_frames[0],
    "stored_vs_rebuilt_vertex_rms":float((rebuilt[0]-stored).square().mean().sqrt()),
    "note":"Recomputed with MeshModel scale=None; all sequence prep_s ranges reported above."}
report["gitignore_unchanged"] = report["gitignore_sha256"] == hashlib.sha256(Path(".gitignore").read_bytes()).hexdigest()
report["peak_cuda_memory_MB"] = torch.cuda.max_memory_allocated()/1024**2 if device.type=="cuda" else None
report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
print("ALL_SEVEN_PILOTS_COMPLETE",report_path,"gitignore_unchanged",report["gitignore_unchanged"],flush=True)

