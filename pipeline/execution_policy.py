"""Reject a GPU-only request before invoking any hybrid analysis component.

This is a static audit of the implementations selected by this pipeline. A
CUDA model device does not make its preprocessing or downstream algorithms CUDA.
Adding a GPU implementation requires updating this audit and its contract tests.
"""


def execution_policy_report(config, *, roster=False, recompute=False):
    policy = config.get("compute_policy", "hybrid")
    if config.get('backend') == 'native_cuda':
        return native_cuda_report(config, roster=roster, recompute=recompute)
    components = []

    def cpu(name, implementation):
        components.append({"name": name, "device": "cpu", "implementation": implementation})

    if not recompute:
        cpu("video_decode", "pipeline.video.Video: OpenCV VideoCapture")
        cpu("people_preprocessing", "Ultralytics / detector NumPy image preprocessing")
        cpu("tracking", "pipeline.adapters.RoleTracker: Ultralytics ByteTrack / BoT-SORT")
        if config["court"]["enabled"]:
            cpu("court_calibration", "training.court.calibration: OpenCV homography, optical flow and projection")
        if config["ball"]["enabled"]:
            cpu("ball_preprocessing", "pipeline.adapters: OpenCV resize and source-coordinate conversion")
        if config["pose"]["enabled"]:
            cpu("pose_preprocessing_and_decoding", "training.players.pose: OpenCV affine crops and NumPy SimCC decoding")
        if config["jersey"]["enabled"]:
            cpu("jersey_crops_and_colors", "training.jersey.crops / colors: OpenCV, Pillow and NumPy")
            cpu("jersey_temporal_votes", "training.jersey.temporal: host vote counters")
        if config["segmentation"]["strategy"] != "disabled":
            cpu("segmentation_mask_encoding", "training.segmentation.masks: NumPy RLE encoding")
        if roster:
            cpu("identity_resolution", "pipeline.identity / jersey_votes / spatial_identity: host votes and geometry")
        cpu("health_statistics", "mlops.health: host counters and ratios")
        for name in ("people", "ball", "court", "pose", "jersey"):
            if name == "people" or config[name]["enabled"]:
                device = config[name]["device"]
                if device == "cpu":
                    cpu(f"{name}_model", f"{name}.device=cpu")
        if config.get("device_fallback") == "cpu":
            cpu("model_fallback", "device_fallback=cpu")
        if config["jersey"]["enabled"] and config["jersey"].get("fallback_device") == "cpu":
            cpu("jersey_fallback", "jersey.fallback_device=cpu")
        if config["pose"]["enabled"] and config["pose"].get("allow_cpu_fallback", True):
            cpu("pose_graph_partitions", "pose.allow_cpu_fallback=true")
    if config["distance"].get("device", "cpu") == "cpu":
        cpu("distance_duration_coverage", "distance.device=cpu")
    # These counters are still on the host even with distance.device=cuda:N.
    cpu("metric_identity_and_exclusion_counts", "pipeline.distance: identity collisions and exclusion Counter")
    return {"compute_policy": policy,
            "audit_scope": "static CPU usage audit; does not validate GPU availability or model execution",
            "compatible": policy != "gpu_only" or not components,
            "cpu_components": components,
            "blocking_components": components if policy == "gpu_only" else []}


def native_cuda_report(config, *, roster=False, recompute=False):
    """Only the implemented, audited native path can satisfy GPU-only."""
    blocking=[]
    def reject(name,reason):blocking.append({'name':name,'reason':reason})
    if config.get('compute_policy')!='gpu_only':reject('compute_policy','native_cuda requires compute_policy=gpu_only')
    if not config['distance'].get('device','cpu').startswith('cuda:'):reject('statistics_device','CUDA required')
    if not recompute:
        if not roster:reject('roster','Native CUDA jersey resolution requires an explicit roster')
        if config['device_fallback'] is not None:reject('device_fallback','CPU fallback must be disabled')
        if config['jersey']['fallback_device'] is not None:reject('jersey_fallback','CPU fallback must be disabled')
        devices={config[name]['device'] for name in ('people','ball','court','pose','jersey','distance')}
        if len(devices)!=1 or not next(iter(devices)).startswith('cuda:'):reject('devices','All components must use the same CUDA device')
        if config['people']['adapter']!='yolo' or config['ball']['adapter']!='yolo':reject('models','Native CUDA currently supports YOLO people and ball only')
        if config['segmentation']['strategy']!='box_prompt':reject('segmentation','Native CUDA requires the pinned SAM2 box-prompt path')
        if not config['segmentation']['segment_referees']:reject('segment_referees','Native CUDA currently segments both players and referees')
        if config['tracking']['tracker']!='bytetrack' or config['tracking']['gmc_method']!='none':reject('tracking','CUDA ByteTrack requires tracker=bytetrack, gmc_method=none')
        if any(not config[name]['enabled'] for name in ('ball','court','pose','jersey')):reject('capabilities','All six capabilities must be enabled for the native path')
        if config['people']['class_roles']!={'2':'player','3':'referee'}:reject('class_roles','Native CUDA requires E-BARD player/referee classes 2/3')
    device=config['distance'].get('device')
    names=('statistics',) if recompute else ('preprocessing','people','ball','pose','segmentation','jersey','tracking','court_calibration',
           'identity_votes','spatial_handoffs','statistics','health')
    return {'compute_policy':config.get('compute_policy'),'backend':'native_cuda',
        'audit_scope':'Native CUDA implementations with runtime TorchDispatch CPU arithmetic rejection; GPU availability and weights checked at startup',
        'compatible':not blocking,'cpu_components':[],'blocking_components':blocking,
        'components':[{'name':name,'device':device} for name in names],
        'decoder':None if recompute else 'NVIDIA_NVDEC','host_scope':'orchestration, container metadata, IDs, JSON and file I/O only'}


def enforce_execution_policy(config, *, roster=False, recompute=False):
    report = execution_policy_report(config, roster=roster, recompute=recompute)
    if not report["compatible"]:
        names = ", ".join(item["name"] for item in report["blocking_components"])
        raise RuntimeError(f"compute_policy=gpu_only is unsupported by the current pipeline; "
                           f"CPU calculations remain in: {names}. Analysis was not started.")
    return report
