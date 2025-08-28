from app.config import settings
import cloudinary, cloudinary.uploader
from cloudinary.utils import cloudinary_url

cloudinary.config(
    cloud_name="dcwak4gp7",
    api_key="522789991142827",
    api_secret="v1COE84XvaqMWmdOY1OffZOWvhc",
    secure=True
)

def init_cloudinary():
    if not settings.CLOUDINARY_URL:
        raise RuntimeError("CLOUDINARY_URL not set")
    cloudinary.config(cloudinary_url=settings.CLOUDINARY_URL)

def upload_image_bytes(img_bytes: bytes, public_id: str) -> tuple[str, str]:
    # returns (original_url, thumb_url)
    init_cloudinary()
    res = cloudinary.uploader.upload(
        img_bytes,
        public_id=public_id,
        overwrite=True,
        resource_type="image",
    )
    orig_url = res.get("secure_url")
    # Build fast thumbnail URL (no extra upload)
    thumb_url, _ = cloudinary_url(
        res.get("public_id"),
        secure=True,
        transformation=[{"width": 640, "crop": "limit", "quality": "auto", "fetch_format": "auto"}],
        format=res.get("format")
    )
    return orig_url, thumb_url
