from lxml import etree
import cv2
import numpy as np  
from lxml import etree,objectify
import os
import shutil 


save_dirpath = '/media/lhpc04/67816e63-c4f9-4450-9981-08f1d435895a/data01/model/UAV_03_B3/images_cut'
image_dirpath = '/media/lhpc04/67816e63-c4f9-4450-9981-08f1d435895a/data01/model/UAV_03_B3/images/'
xml_dirpath = '/media/lhpc04/67816e63-c4f9-4450-9981-08f1d435895a/data01/model/UAV_03_B3/xmls/'

save_image_dirpath = os.path.join(save_dirpath, 'images')
save_xml_dirpath = os.path.join(save_dirpath, 'xmls')

show_label = ['signal','2']

label_ext=['1','2','3','4','5','6','7','8','9','10','11','12']
label_small=['OTHER', 'P', 'PM']
train_side = 504
image_ext = ['jpg', 'jpeg', 'bmp', 'png', 'tiff', 'gif', 'pcx', 'tga', 'exif', 'fpx', 'svg', 'psd', 'cdr', 'pcd', 'dxf', 'ufo', 'eps', 'ai', 'raw', 'wmf', 'webp']
label_name=[]
def load_axis(obj):
    xmin_t = int(obj.xpath('.//xmin')[0].text)
    xmax_t = int(obj.xpath('.//xmax')[0].text)
    ymin_t = int(obj.xpath('.//ymin')[0].text)
    ymax_t = int(obj.xpath('.//ymax')[0].text)

    xmin = min(xmax_t, xmin_t)
    xmax = max(xmax_t, xmin_t)
    ymin = min(ymax_t, ymin_t)
    ymax = max(ymax_t, ymin_t)

    return int(xmin),int(xmax),int(ymin),int(ymax)
def load_xml_file(xml_file_name):
    # print(xml_file_name)
    file_info = {}
    boxs = [] 
    xml = etree.parse(xml_file_name)
    root = xml.getroot()
    objects = root.xpath('//object')
    file_info['size']=[root.xpath('//size/width')[0].text,root.xpath('//size/height')[0].text,root.xpath('//size/depth')[0].text]
    file_info['folder']=root.xpath('folder')[0].text
    file_info['filename']=root.xpath('filename')[0].text
    file_info['path']=root.xpath('path')[0].text

    for obj in objects:
        obj_name = obj.xpath('.//name')[0].text
        if (obj_name in label_name) or (len(label_name) == 0):
            xmin,xmax,ymin,ymax = load_axis(obj)
            boxs.append([obj_name,xmin,xmax,ymin,ymax])
    file_info['boxs'] = boxs
    return file_info

# 保存节点
def save_node(anno_tree, obj_info):
    
    e = objectify.ElementMaker(annotate=False)
    e2 = objectify.ElementMaker(annotate=False)
    anno_tree2 = e2.object(
        e.name(obj_info[0]),
        e.pose('Unspecified'),
        e.truncated(0),
        e.difficult(0),
        e.bndbox(
            e.xmin(int(obj_info[1])),
            e.xmax(int(obj_info[2])),
            e.ymin(int(obj_info[3])),
            e.ymax(int(obj_info[4]))
        )
    )
    anno_tree.append(anno_tree2)

def save_xmlfile(xml_info, file_name):
    # 文件信息
    e = objectify.ElementMaker(annotate=False)
    anno_tree = e.annotation(
        e.folder(xml_info['folder']),
        e.filename(xml_info['filename']),
        e.path(xml_info['path']),
        e.source(
            e.database('Unknown')
        ),
        e.size(
            e.width(xml_info['size'][0]),
            e.height(xml_info['size'][1]),
            e.depth(xml_info['size'][2]),
        ),
        e.segmented(0),
    )
    index = 0
    for box in xml_info['boxs']:
        save_node(anno_tree, box)
        index += 1
    if index > 0:
        etree.ElementTree(anno_tree).write(file_name, pretty_print=True)   

# 去掉父路径，得到相对路径
def dup_papath(path, parent_path):
    if path == None:
        return None
    if parent_path == None:
        return path 
    if len(path) < len(parent_path):
        return None
    elif path == parent_path:
        return '.'

    if path[:len(parent_path)] == parent_path:
        return path[len(parent_path):].strip().strip('/').strip('\\')

# 区分大小,第0维标记大边的维数， 0表示宽为大边，1表示高为大边 
# 注意： 由于输入模型的图片一般都会被 压缩 为正方形， 所以这里的大小边比较时要加 宽高比
def distinguish_side(proportion, boxs):
    side_dist = []
    for box in boxs: 
        width = abs(box[2]-box[1])
        height = abs(box[4]-box[3])
        dist = [0, width, height] if (width)*proportion > (height) else [1, width, height] 
        side_dist.append(box+[dist])
    return side_dist
#coding:utf-8  

def boxs2array(boxs):
    temp = []
    for box in boxs:
        temp.append(box[1:])
    return np.array(temp)*1.0

def ext_box_adjust(ext_boxs, shape):
    image_width = int(shape[0])
    image_height = int(shape[1])
    pro_hw = image_height/image_width
    adjust_box = []
    for ext_box in ext_boxs:
        width = ext_box[2] -ext_box[1]
        height = ext_box[4] -ext_box[3]
        if pro_hw < 1:
            min_side = max(int(width*pro_hw), height, train_side)
        else:
            min_side = max(width, int(height/pro_hw), train_side)
        if pro_hw < 1:
            height_t = min_side
            width_t = int(min_side/pro_hw)
        else:
            height_t = int(min_side*pro_hw)
            width_t = min_side
        height_e = (height_t-height)/2
        width_e = (width_t-width)/2
        
        # 当截框的边超出了大图，往另外的一个方向偏移
        x1 = max(0, int(max(0, ext_box[1] - width_e) - max(0, ext_box[2] + width_e - image_width)))
        x2 = min(image_width, int(min(image_width, ext_box[2] + width_e) - min(0, ext_box[1] - width_e)))
        y1 = max(0, int(max(0, ext_box[3] - height_e) - max(0, ext_box[4] + height_e - image_height)))
        y2 = min(image_height,int(min(image_height, ext_box[4] + height_e) - min(0, ext_box[3] - height_e)))

        adjust_box.append([ext_box[0],x1,x2,y1,y2])
    return adjust_box

def get_extbox(xml_path, image_path):
    print(image_path)
    # 获取所有标签框
    xml_info = load_xml_file(xml_path)
    xml_big_info = {}
    # print(xml_info)
    #复制一份不包含小目标, 并将小目标剔出来
    small_boxs = []
    ext_boxs = []
    for key,value in xml_info.items():
        if key == 'boxs':
            xml_big_info[key] = []
            for box in value:
                if box[0] in label_small: #另存小目标, 不保存到原图
                    small_boxs.append(box)
                    continue
                if box[0] in label_ext: # 另存截图框, 保存到原图
                    ext_boxs.append(box)
                xml_big_info[key].append(box)
        else:
            xml_big_info[key] = value

    save_xml_path = os.path.join(save_xml_dirpath, dup_papath(xml_path, xml_dirpath))
    save_image_path = os.path.join(save_image_dirpath, dup_papath(image_path, image_dirpath))
    if not os.path.exists(os.path.dirname(save_xml_path)):
        os.makedirs(os.path.dirname(save_xml_path))
    if not os.path.exists(os.path.dirname(save_image_path)):
        os.makedirs(os.path.dirname(save_image_path))
    shutil.copy(image_path, save_image_path)
    save_xmlfile(xml_big_info, save_xml_path)  
    if len(ext_boxs) == 0:
        shutil.copy(xml_path, save_xml_path)
    small_boxs_np = boxs2array(xml_info['boxs'])#[small_boxs[0],np.array(small_boxs[1:][:])]
    # print("xml_info['boxs']", xml_info['boxs'])
    # print("small_boxs", small_boxs)
    
    # 通过 扩展框 截图 和 保存xml信息
    adjust_box = ext_box_adjust(ext_boxs, xml_info['size'])
    ext_boxs_np = boxs2array(adjust_box)
    for ext_i, ext_box in enumerate(ext_boxs_np):
        xx1 = np.maximum(ext_box[0], small_boxs_np[:,0])  
        xx2 = np.minimum(ext_box[1], small_boxs_np[:,1])  
        yy1 = np.maximum(ext_box[2], small_boxs_np[:,2])  
        yy2 = np.minimum(ext_box[3], small_boxs_np[:,3])  

        w = np.maximum(0.0, xx2 - xx1)  
        h = np.maximum(0.0, yy2 - yy1)  
        inter = w * h  
        ovr = (inter*1.0) / (((small_boxs_np[:,1]-small_boxs_np[:,0]) * (small_boxs_np[:,3]-small_boxs_np[:,2]))*1.0)

        inds = np.where(ovr >= 1)[0] 
        coordinate_t = np.array([int(ext_box[0]),int(ext_box[0]),int(ext_box[2]),int(ext_box[2])])
        cur_cut = []
        cut_coordinate = small_boxs_np-coordinate_t
        for i in inds:
            if xml_info['boxs'][i][0] in label_ext:
                continue
            cur_cut.append([xml_info['boxs'][i][0]]+cut_coordinate[i].tolist())
        if len(inds) > 0:
            image = cv2.imread(image_path)
            cur_info = {}
            cur_info['size'] = [int(ext_box[1]-ext_box[0]), int(ext_box[3]-ext_box[2]), 3]
            cur_info['folder'] = os.path.dirname(image_path)
            cur_info['filename'] = os.path.basename(image_path)
            cur_info['path'] = image_path
            cur_info['boxs'] = cur_cut
            split_name = os.path.splitext(save_xml_path)
            save_xmlfile(cur_info, split_name[0]+'_'+str(ext_i)+split_name[1])
            split_name = os.path.splitext(save_image_path)
            cv2.imwrite(split_name[0]+'_'+str(ext_i)+split_name[1], image[int(ext_box[2]):int(ext_box[3]), int(ext_box[0]):int(ext_box[1])])


def list_dir():
    try:
        image_dirs = os.listdir(image_dirpath)
    except Exception as e:
        print(e)
        return 
    for image_dir in image_dirs:
        try:
            image_names = os.listdir(os.path.join(image_dirpath,image_dir))
        except Exception as e:
            print(e)
            continue
        for image_name in image_names:
            split_name = os.path.splitext(image_name)
            if not (split_name[1][1:].lower() in image_ext):
                continue
            # print(image_name)
            image_path = os.path.join(image_dirpath, image_dir, image_name)
            relate_image_path = dup_papath(image_path, image_dirpath)
            split_relate = os.path.splitext(relate_image_path)
            xml_path = os.path.join(xml_dirpath, split_relate[0]+'.xml')
            get_extbox(xml_path, image_path)

list_dir()
    
