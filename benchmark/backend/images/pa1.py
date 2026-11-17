"""PA1 feature layout and inspectable preprocessing; no learned parameters."""
import io

import numpy as np
from PIL import Image

DIMENSIONS={'hsv':128,'lbp':10,'glcm':48,'hog':1764,'canny':64}
TITLES={'hsv':'Màu sắc HSV','lbp':'Kết cấu LBP','glcm':'Đồng xuất hiện GLCM','hog':'Hướng gradient HOG','canny':'Biên Canny'}


def blocks(vector,config):
    values=np.asarray(vector,dtype=np.float32)
    offset=0
    total=float(np.dot(values,values))
    result=[]
    for group in config.get('groups',list(DIMENSIONS)):
        end=offset+DIMENSIONS[group]
        part=values[offset:end]
        result.append(dict(group=group,name=TITLES[group],start=offset,end_exclusive=end,dimension=DIMENSIONS[group],
                           weight=config.get('weights',{}).get(group,1),l2_norm=float(np.linalg.norm(part)),
                           energy_share=float(np.dot(part,part))/total if total else 0))
        offset=end
    if offset!=len(values):
        raise ValueError('PA1 feature layout does not match vector dimension')
    return result


def explain(query,document,config):
    q=np.asarray(query,dtype=np.float32)
    d=np.asarray(document,dtype=np.float32)
    if q.shape!=d.shape:
        raise ValueError('PA1 query/document feature dimensions differ')
    result=[]
    for block in blocks(q,config):
        start,end=block['start'],block['end_exclusive']
        a,b=q[start:end],d[start:end]
        contribution=float(np.dot(a,b))
        norm=float(np.linalg.norm(a)*np.linalg.norm(b))
        result.append(dict(group=block['group'],name=block['name'],contribution=contribution,
                           cosine=contribution/norm if norm>1e-12 else None,weight=block['weight']))
    return result


def preview(image,kind):
    import cv2
    from skimage.feature import local_binary_pattern,graycomatrix,hog
    from skimage.exposure import rescale_intensity
    rgb=np.asarray(image.resize((128,128),Image.Resampling.BICUBIC))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    if kind=='rgb':
        output=rgb
    elif kind=='gray':
        output=gray
    elif kind=='lbp':
        output=(local_binary_pattern(gray,8,1,method='uniform')/9*255).astype(np.uint8)
    elif kind=='canny':
        output=cv2.Canny(gray,100,200)
    elif kind=='hog':
        _,view=hog(gray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm='L2-Hys',visualize=True)
        output=rescale_intensity(view,out_range=(0,255)).astype(np.uint8)
    elif kind=='glcm':
        matrix=graycomatrix(gray//16,distances=[1,3],angles=[0,np.pi/4,np.pi/2,3*np.pi/4],levels=16,symmetric=True,normed=True).mean(axis=(2,3))
        output=cv2.applyColorMap(rescale_intensity(matrix,out_range=(0,255)).astype(np.uint8),cv2.COLORMAP_VIRIDIS)[:,:,::-1]
    else:
        raise ValueError('Unknown PA1 preview')
    target=Image.fromarray(output).resize((256,256),Image.Resampling.NEAREST)
    buffer=io.BytesIO()
    target.save(buffer,format='PNG')
    return buffer.getvalue()
