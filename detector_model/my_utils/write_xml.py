import os
import xml.dom.minidom


# https://www.cnblogs.com/zhangxingcomeon/p/15993512.html
def write_xml(imgs_folder: str, img_name: str, img_path: str, xml_folder: str, img_width: int, img_height: int,
              tag_names: list, boxes: list, ref=False):
    '''
    VOC标注xml文件生成函数
    :param xml_folder:
    :param imgs_folder: 文件夹名
    :param img_name:
    :param img_path:
    :param img_width:
    :param img_height:
    :param tag_names:
    :param boxes: [[xmin1, ymin1, xmax1, ymax1],[xmin2, ymin2, xmax2, ymax2]....]
    :return: a standard VOC format .xml file, named "img_name.xml"
    '''
    # 创建dom树对象
    doc = xml.dom.minidom.Document()

    # 创建root结点annotation，并用dom对象添加根结点
    root_node = doc.createElement("annotation")
    doc.appendChild(root_node)

    # 创建结点并加入到根结点
    folder_node = doc.createElement("folder")
    folder_value = doc.createTextNode(imgs_folder)
    folder_node.appendChild(folder_value)
    root_node.appendChild(folder_node)

    filename_node = doc.createElement("filename")
    filename_value = doc.createTextNode(img_name)
    filename_node.appendChild(filename_value)
    root_node.appendChild(filename_node)

    path_node = doc.createElement("path")
    path_value = doc.createTextNode(img_path)
    path_node.appendChild(path_value)
    root_node.appendChild(path_node)

    source_node = doc.createElement("source")
    database_node = doc.createElement("database")
    database_node.appendChild(doc.createTextNode("Unknown"))
    source_node.appendChild(database_node)
    root_node.appendChild(source_node)

    size_node = doc.createElement("size")
    for item, value in zip(["width", "height", "depth"], [img_width, img_height, 3]):
        elem = doc.createElement(item)
        elem.appendChild(doc.createTextNode(str(value)))
        size_node.appendChild(elem)
    root_node.appendChild(size_node)

    seg_node = doc.createElement("segmented")
    seg_node.appendChild(doc.createTextNode(str(0)))
    root_node.appendChild(seg_node)

    if ref:
        temp = []
        for i, bbox in enumerate(boxes):
            xmin, ymin, xmax, ymax = bbox
            xmin *= img_width - 1
            xmax *= img_width - 1
            ymin *= img_height - 1
            ymax *= img_height - 1
            temp.append((int(xmin), int(ymin), int(xmax), int(ymax)))
        boxes = temp

    for i, tag_name in enumerate(tag_names):
        obj_node = doc.createElement("object")
        name_node = doc.createElement("name")
        name_node.appendChild(doc.createTextNode(tag_name))
        obj_node.appendChild(name_node)

        pose_node = doc.createElement("pose")
        pose_node.appendChild(doc.createTextNode("Unspecified"))
        obj_node.appendChild(pose_node)

        trun_node = doc.createElement("truncated")
        trun_node.appendChild(doc.createTextNode(str(0)))
        obj_node.appendChild(trun_node)

        trun_node = doc.createElement("difficult")
        trun_node.appendChild(doc.createTextNode(str(0)))
        obj_node.appendChild(trun_node)

        bndbox_node = doc.createElement("bndbox")
        for item, value in zip(["xmin", "ymin", "xmax", "ymax"], boxes[i]):
            elem = doc.createElement(item)
            elem.appendChild(doc.createTextNode(str(value)))
            bndbox_node.appendChild(elem)
        obj_node.appendChild(bndbox_node)
        root_node.appendChild(obj_node)

    # print(img_name.split('.')[-2] + ".xml")
    file_path = os.path.join(xml_folder, img_name.split('.')[-2] + ".xml")
    with open(file_path, "w", encoding="utf-8") as f:
        # writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
        # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")


if __name__ == '__main__':
    write_xml(imgs_folder=r'imgs', img_name='test.png',
              img_path=r'E:\my_files\programmes\python\dp_dataset',
              xml_folder=r'E:\my_files\programmes\python\dp_dataset',
              img_width=640, img_height=480,
              tag_names=['person'], boxes=[(1, 2, 3, 4)])
