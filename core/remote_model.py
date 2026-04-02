"""
Women Safety AI — Remote Model Client
======================================
Utility for calling remote model inference API from the backend.

Usage:
    from core.remote_model import RemoteModelClient
    
    client = RemoteModelClient(
        api_url="https://model.yourdomain.com",
        timeout=30
    )
    
    detections = await client.detect(image)
    gender = await client.classify_gender(person_crop)
"""

import aiohttp
import base64
import cv2
import numpy as np
from typing import Optional, List, Dict, Any


class RemoteModelClient:
    """Client for calling remote model inference server."""

    def __init__(self, api_url: str, timeout: int = 30):
        """
        Initialize remote model client.

        Args:
            api_url: Base URL of remote model server (e.g., https://model.yourdomain.com)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Create aiohttp session if needed."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()

    @staticmethod
    def _image_to_base64(image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string."""
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    async def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Detect persons in image.

        Args:
            image: OpenCV image (BGR)
            confidence_threshold: Detection confidence threshold

        Returns:
            List of detections with boxes and confidence scores
        """
        await self._ensure_session()

        image_b64 = self._image_to_base64(image)

        try:
            async with self.session.post(
                f"{self.api_url}/api/v1/detect",
                json={
                    "image_base64": image_b64,
                    "confidence_threshold": confidence_threshold,
                },
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Model server error: {resp.status}")
                
                result = await resp.json()
                if result.get("status") != "success":
                    raise RuntimeError(f"Inference failed: {result.get('error')}")
                
                return result.get("data", {}).get("detections", [])

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to model server: {e}")

    async def classify_gender(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify gender from person image.

        Args:
            image: OpenCV image (BGR)

        Returns:
            Gender classification with confidence
        """
        await self._ensure_session()

        image_b64 = self._image_to_base64(image)

        try:
            async with self.session.post(
                f"{self.api_url}/api/v1/gender",
                json={"image_base64": image_b64},
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Model server error: {resp.status}")
                
                result = await resp.json()
                if result.get("status") != "success":
                    raise RuntimeError(f"Inference failed: {result.get('error')}")
                
                return result.get("data", {})

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to model server: {e}")

    async def estimate_pose(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Estimate pose keypoints from image.

        Args:
            image: OpenCV image (BGR)

        Returns:
            List of pose estimations with keypoints
        """
        await self._ensure_session()

        image_b64 = self._image_to_base64(image)

        try:
            async with self.session.post(
                f"{self.api_url}/api/v1/pose",
                json={"image_base64": image_b64},
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Model server error: {resp.status}")
                
                result = await resp.json()
                if result.get("status") != "success":
                    raise RuntimeError(f"Inference failed: {result.get('error')}")
                
                return result.get("data", {}).get("poses", [])

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to model server: {e}")

    async def detect_assault(self, image: np.ndarray) -> float:
        """
        Detect assault/violence in image.

        Args:
            image: OpenCV image (BGR)

        Returns:
            Assault detection score (0-1)
        """
        await self._ensure_session()

        image_b64 = self._image_to_base64(image)

        try:
            async with self.session.post(
                f"{self.api_url}/api/v1/assault",
                json={"image_base64": image_b64},
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Model server error: {resp.status}")
                
                result = await resp.json()
                if result.get("status") != "success":
                    raise RuntimeError(f"Inference failed: {result.get('error')}")
                
                return float(result.get("data", {}).get("assault_score", 0.0))

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to model server: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if model server is alive and which models are loaded.

        Returns:
            Health check response
        """
        await self._ensure_session()

        try:
            async with self.session.post(f"{self.api_url}/api/v1/health") as resp:
                if resp.status != 200:
                    return {"status": "error", "error": f"HTTP {resp.status}"}
                return await resp.json()

        except aiohttp.ClientError as e:
            return {"status": "error", "error": str(e)}


# Example usage for testing
if __name__ == "__main__":
    import asyncio

    async def test():
        client = RemoteModelClient("http://localhost:9000", timeout=10)

        try:
            # Check health
            health = await client.health_check()
            print("Health:", health)

            # Load test image
            test_image = cv2.imread("test.jpg")
            if test_image is not None:
                # Run detection
                detections = await client.detect(test_image)
                print("Detections:", detections)

        finally:
            await client.close()

    asyncio.run(test())
