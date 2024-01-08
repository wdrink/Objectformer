import os

def imd20():
    root = "/data1/v-junkewang/TamperingData/imd20"
    au_dir = "Au"
    tp_dir = "Sp"

    au_path = os.path.join(root, au_dir)
    tp_path = os.path.join(root, tp_dir)

    with open(os.path.join(root, "imd20.txt"), "w") as f:
        au_images = os.listdir(au_path)
        for img in au_images:
            item = os.path.join("imd20", "Au", img) + "\t" + "None" + "\t" + "0"
            f.write(item + "\n")
        
        print("Au: ", len(au_images))
        tp_images = os.listdir(tp_path)
        suc = 0
        for img in tp_images:
            mask_path = os.path.join("imd20", "Mask", os.path.basename(img).split(".")[0] + "_mask.png")
            
            if os.path.exists(os.path.join(os.path.dirname(root), mask_path)):
                item = os.path.join("imd20", "Sp", img) + "\t" + mask_path + "\t" + "1"
                suc += 1
                f.write(item + "\n")
            
            else:
                # print(mask_path)
                continue
        
        print("Tp: ", suc)



def casia():
    root = "/data1/v-junkewang/TamperingData/Casia/casia/CASIA1"
    au_dir = "Au"
    tp_dir = "Sp"

    au_path = os.path.join(root, au_dir)
    tp_path = os.path.join(root, tp_dir)

    with open(os.path.join(root, "CASIAv1.txt"), "w") as f:
        au_images = os.listdir(au_path)
        for img in au_images:
            item = os.path.join("Casia/casia/CASIA1", "Au", img) + "\t" + "None" + "\t" + "0"
            f.write(item + "\n")
        
        print("Au: ", len(au_images))
        tp_images = os.listdir(tp_path)
        suc = 0
        for img in tp_images:
            mask_path = os.path.join("Casia/casia/CASIA1", "mask", os.path.basename(img).split(".")[0] + "_gt.png")
            
            if os.path.exists(os.path.join("/data1/v-junkewang/TamperingData/", mask_path)):
                item = os.path.join("Casia/casia/CASIA1", "Sp", img) + "\t" + mask_path + "\t" + "1"
                suc += 1
                f.write(item + "\n")
            
            else:
                # print(mask_path)
                continue
        
        print("Tp: ", suc)



def coverage():
    root = "/data1/v-junkewang/TamperingData/Coverage"
    au_dir = "image"
    tp_dir = "image"

    au_path = os.path.join(root, au_dir)
    tp_path = os.path.join(root, tp_dir)

    with open(os.path.join(root, "coverage.txt"), "w") as f:
        au_images = os.listdir(au_path)
        for img in au_images:
            if "t." in img:
                continue
            item = os.path.join("Coverage", "image", img) + "\t" + "None" + "\t" + "0"
            f.write(item + "\n")
        
        print("Au: ", len(au_images))
        tp_images = os.listdir(tp_path)
        suc = 0
        for img in tp_images:
            if "t." not in img:
                continue
            mask_path = os.path.join("Coverage", "mask", os.path.basename(img).split(".")[0][:-1] + "forged.tif")
            if os.path.exists(os.path.join(os.path.dirname(root), mask_path)):
                item = os.path.join("Coverage", "image", img) + "\t" + mask_path + "\t" + "1"
                suc += 1
                f.write(item + "\n")
            
            else:
                # print(mask_path)
                continue
        
        print("Tp: ", suc)



def columbia():
    root = "/data1/v-junkewang/TamperingData/Columbia"
    au_dir = "4cam_auth"
    tp_dir = "4cam_splc"

    au_path = os.path.join(root, au_dir)
    tp_path = os.path.join(root, tp_dir)

    with open(os.path.join(root, "columbia.txt"), "w") as f:
        au_images = os.listdir(au_path)
        for img in au_images:
            item = os.path.join("Columbia", "4cam_auth", img) + "\t" + "None" + "\t" + "0"
            f.write(item + "\n")
        
        print("Au: ", len(au_images))
        tp_images = os.listdir(tp_path)
        suc = 0
        for img in tp_images:
            mask_path = os.path.join("Columbia", "4cam_splc/edgemask", os.path.basename(img).split(".")[0] + "_edgemask.jpg")
            
            if os.path.exists(os.path.join(os.path.dirname(root), mask_path)):
                item = os.path.join("Columbia", "4cam_splc", img) + "\t" + mask_path + "\t" + "1"
                suc += 1
                f.write(item + "\n")
            
            else:
                # print(mask_path)
                continue
        
        print("Tp: ", suc)



casia()