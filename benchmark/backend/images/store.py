import hashlib
import io
import os
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .. import data

MAX_BYTES = int(os.getenv('IMAGE_MAX_UPLOAD_MB', '20')) * 1024 * 1024
MAX_PIXELS = int(os.getenv('IMAGE_MAX_PIXELS', '25000000'))


def decode(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError(f'Ảnh rỗng hoặc vượt {MAX_BYTES//1024//1024} MB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in ('JPEG','PNG','WEBP'):
                    raise ValueError('Chỉ hỗ trợ JPEG, PNG, WEBP')
                if source.width * source.height > MAX_PIXELS or min(source.size) < 16:
                    raise ValueError('Ảnh phải ≥16 px mỗi cạnh và không vượt giới hạn pixel')
                source.load()
                # Defined alpha behavior, EXIF orientation and color conversion for every method.
                image = ImageOps.exif_transpose(source).convert('RGBA')
                background = Image.new('RGBA', image.size, 'white')
                return Image.alpha_composite(background,image).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Ảnh không giải mã được hoặc vượt giới hạn an toàn') from exc


def register(raw, place_id='', name='', dataset_id=None):
    image = decode(raw)
    pixel_hash = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    image_id = pixel_hash[:32]
    if dataset_id:
        dataset = data.get('datasets', dataset_id)
        place = next((p for p in dataset['records'] if p['place_id']==place_id),None)
        if dataset['kind']!='places' or not place:
            raise ValueError('place_id không có trong dataset đã chọn')
        name = place['name']
    with data.LOCK:
        try:
            old = data.get('image_assets',image_id)
            if old['place_id']!=place_id or old.get('dataset_id')!=dataset_id:
                raise ValueError('Ảnh trùng nội dung đã liên kết với place/dataset khác')
            return old
        except FileNotFoundError:
            pass
        folder = data.ROOT / 'image_blobs'
        folder.mkdir(exist_ok=True,parents=True)
        image.save(folder/(image_id+'.png'),format='PNG')
        value = dict(id=image_id,created_at=data.now(),place_id=place_id,name=name or place_id,
                     dataset_id=dataset_id,width=image.width,height=image.height,pixel_sha256=pixel_hash,
                     source_sha256=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw))
        data.save('image_assets',image_id,value)
        return value


def load(image_id):
    data.get('image_assets',image_id)  # validate ID before constructing the path
    with Image.open(data.ROOT/'image_blobs'/(image_id+'.png')) as image:
        return image.convert('RGB')
