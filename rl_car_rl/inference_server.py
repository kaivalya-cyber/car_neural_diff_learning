"""
REST inference server for deploying trained policies as HTTP APIs.

Usage:
    python inference_server.py --checkpoint checkpoints/best.pth --port 8000
    python inference_server.py --checkpoint checkpoints/best.pth --onnx --port 8000

Endpoints:
    POST /predict   — Accept observation, return action
    GET  /health    — Server health check
    POST /batch     — Accept batch of observations, return batch of actions
"""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class ObservationRequest(BaseModel):
    observation: list[float]
    deterministic: bool = True


class BatchRequest(BaseModel):
    observations: list[list[float]]
    deterministic: bool = True


class ActionResponse(BaseModel):
    steering: float
    throttle: float


class BatchResponse(BaseModel):
    actions: list[ActionResponse]


def create_inference_app(checkpoint_path: str, state_dim: int = 20, use_onnx: bool = False):
    """Create and configure the FastAPI inference application."""
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI and uvicorn are required. Install with: pip install fastapi uvicorn")
        return None

    app = FastAPI(title="RL Car Inference Server", version="1.0.0")

    # Load model
    if use_onnx:
        from export import ONNXInferenceRunner
        model = ONNXInferenceRunner(checkpoint_path)
        print(f"Loaded ONNX model from {checkpoint_path}")
    else:
        import torch
        from export import ExportedPolicy

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        log_std = checkpoint.get("policy_log_std")
        action_dim = 2 if log_std is None else log_std.shape[-1]

        policy_weights = checkpoint.get("policy_net", {})
        hidden_size = 256
        for key in policy_weights:
            if "weight" in key:
                hidden_size = policy_weights[key].shape[0]
                break

        model = ExportedPolicy(state_dim, action_dim, hidden_size=hidden_size)
        model.net.load_state_dict(policy_weights)
        model.eval()
        print(f"Loaded TorchScript model from {checkpoint_path} (hidden={hidden_size})")

    @app.get("/health")
    async def health():
        return {"status": "ok", "checkpoint": checkpoint_path, "onnx": use_onnx}

    @app.post("/predict", response_model=ActionResponse)
    async def predict(req: ObservationRequest):
        try:
            obs = np.array(req.observation, dtype=np.float32)
            if len(obs) != state_dim:
                raise HTTPException(400, f"Expected {state_dim} dims, got {len(obs)}")

            if use_onnx:
                action = model.predict(obs)
            else:
                import torch
                with torch.no_grad():
                    x = torch.tensor(obs).unsqueeze(0)
                    steering, throttle = model(x)
                    action = np.concatenate([steering.numpy(), throttle.numpy()], axis=-1)[0]

            return ActionResponse(steering=float(action[0]), throttle=float(action[1]))
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/batch", response_model=BatchResponse)
    async def batch_predict(req: BatchRequest):
        try:
            obs = np.array(req.observations, dtype=np.float32)
            if obs.shape[1] != state_dim:
                raise HTTPException(400, f"Expected {state_dim} dims per obs, got {obs.shape[1]}")

            if use_onnx:
                actions = model.predict(obs)
            else:
                import torch
                with torch.no_grad():
                    x = torch.tensor(obs)
                    steering, throttle = model(x)
                    actions = np.concatenate([steering.numpy(), throttle.numpy()], axis=-1)

            result = [ActionResponse(steering=float(a[0]), throttle=float(a[1])) for a in actions]
            return BatchResponse(actions=result)
        except Exception as e:
            raise HTTPException(500, str(e))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REST inference server for RL car policy")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="Path to checkpoint")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--state-dim", type=int, default=20, help="Observation dimension")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model instead of TorchScript")
    args = parser.parse_args()

    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is required. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    app = create_inference_app(args.checkpoint, args.state_dim, args.onnx)
    if app:
        print(f"Inference server starting at http://{args.host}:{args.port}")
        print(f"  Endpoints: /predict (POST), /batch (POST), /health (GET)")
        print(f"  API docs:  http://{args.host}:{args.port}/docs")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
