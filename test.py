
import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def my_ap_per_class(tp, conf, pred_cls, target_cls,target_ids):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
        label_ids: Label ids of the targets (nparray)
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    _ ,tMask = np.unique(target_ids, return_index=True)
    target_cls, target_ids = target_cls[tMask],target_ids[tMask]
    
    labels = ["Buildings" ,"Small Aircraft", 
          "Large aircraft","Vehicles","Bus/Trucks","Boat"]
    colors = ['b','g','r','g','m','y']

        
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    s = [len(unique_classes), tp.shape[1]]  
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    plt.figure(dpi=80)
    print("{:15},{:10},{:10},{:10},{:10},{:10}".format("label","    n_gt","  n_preds",
                                                 "    tp","     fp","  mAP@0.25"))
    plt.figure(dpi=200)
    for ci, c in enumerate(unique_classes):
        mask = (target_cls == c)
        i = pred_cls == c
        n_gt_mask = (target_cls == c)  # Number of ground truth objects
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects
            
        
        if n_p == 0 or n_gt == 0:
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            
            tp_sum = tp[i].sum(0)
            fp_sum = (1 - tp[i]).sum(0)
            print("{:15},{:10},{:10},{:10},{:10},{:10}".format(labels[int(c)],0,len(tp[i]),
                                                               int(tp_sum),int(fp_sum),
                                                               0))
            p[ci] = 0
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            
            tp_sum = tp[i].sum(0)
            fp_sum = (1 - tp[i]).sum(0)
            print("{:15},{:10},{:10},{:10},{:10},{:10}".format(labels[int(c)],n_gt,len(tp[i]),
                                                               int(tp_sum),int(fp_sum),
                                                               str( round(float(tp_sum/(tp_sum+fp_sum)),2) ) ))
            
            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = recall[-1]
            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = precision[-1]
            
            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
                pass
            
            # recall = np.append(recall,recall[-1])
            # precision = np.append(precision,0)
            # Plot            
            plt.plot(recall,precision,label=labels[int(c)],c=colors[int(c)])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title("model performance")
            plt.xlim(0,1)
            plt.xlim(0,1)
            pass
        pass


    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    plt.legend()
    plt.show()
    
    return ((p, r, ap, f1, unique_classes.astype('int32') ) , plt )


def my_test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         conf_thres=0.001,
         iou_thres=0.5,  # for nms
         save_json=False,
         single_cls=False,
         model=None,
         dataloader=None):

    device = torch_utils.select_device('', batch_size=batch_size)
    verbose = True
    # Remove previous
    for f in glob.glob('test_batch*.jpg'):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg, img_size).to(device)
    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    #path = data['test']  # path to test images
    names = load_classes(data['names'])  # class names
    #iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = torch.linspace(0.25, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()
    
    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, img_size, batch_size, rect=True)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), \
                                                 batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.25', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class, allLabelIds = [], [], [], [],[]
    
    detectedIds = [] # target unique ids
    interDict = {}
    unionDict = {}
    d_ids = {}
    
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        _, _, height, width = imgs.shape # batch size, channels, height, width
        
        # Plot images with bounding boxe
        if batch_i == 0 and not os.path.exists('test_batch0.png'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.png')
            pass
        
        # Disable gradients
        with torch.no_grad():
            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs
            
            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls
                pass
            # Run NMS
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            pass
        

        # Statistics per image
        for si, pred in enumerate(output):
            # keep track of correctly detected targets in image, and remove
            toRemove = []
            toRemoveGt = []
            
            mask = (targets[:, 0] == si)
            labels = targets[mask, 1:] # get labels for category
            
            nl = len(labels)
            tids = labels[:,5].tolist() if nl else [] # target id
            tcls = labels[:,0].tolist() if nl else []  # target class
            seen += 1
            
            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                  torch.Tensor(), torch.Tensor(), tcls,tids))
                    allLabelIds.append([-1])
                continue
            
            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]
            
            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78],
                # "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                # to original shape
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(d[5])],
                                  'bbox': [floatn(x, 3) for x in box[di]],
                                  'score': floatn(d[4], 5)})
                    pass
                pass

            # Assign all predictions as incorrect
            correct = torch.zeros(len(pred), niou, dtype=torch.bool)
            label_ids = torch.zeros(len(pred), niou, dtype=int)
            
            if nl:
                detected = []  # target indices
                detectedFracs = {}                
                
                tcls_tensor = labels[:, (0,5)]
                # target boxes
                tbox = \
                    xywh2xyxy(labels[:, 1:5]) * torch.Tensor([width, height,
                                                              width, height]).to(device)
                
                # Per target class
                for cls in torch.unique(tcls_tensor):
                    mask = (cls == tcls_tensor[:,0])
                    ti = mask.nonzero().view(-1)  # prediction indices
                    ids= tcls_tensor[mask,1].view(-1) # class ids
                    pi = (cls == pred[:, 5]).nonzero().view(-1) #target indices
                    # Search for detections
                    if len(pi):
                        # Prediction to target ious
                        ious,inter,union = box_iou(pred[pi, :4], tbox[ti])
                        ious_ = ious
                        ious, i = ious.max(1)  # best ious rowwise, and idx
                        #print(ious)
                        for k in(ious <= iouv[0]).nonzero():
                            rowIdx = i[k] # best iou by column
                            did = tids[rowIdx]                            
                            d_ids[did] = (k, i[k], 0,0)
                            pass
                        
                        for k in(ious > iouv[0]).nonzero():
                            rowIdx = i[k] # best iou by column
                            did = tids[rowIdx]

                            d = ti[rowIdx]  # detected targets
                            
                            numerator = inter[k,i[k]]
                            denom = union[k,i[k]]
                            
                            interDict[did] = interDict.get(did,0)+numerator
                            unionDict[did] = unionDict.get(did,0)+denom
                            # If not detected already
                            if d not in detected:
                                detected.append(d) # append detected target
                                if did not in detectedIds:
                                    # Mask of detections over thresohld(bool)
                                    mask = (ious[k] > iouv).cpu()
                                    detectedIds.append(did)
                                    pass
                                elif did in detectedIds:
                                    # GT Target already detected, check cumulative iou
                                    (k_, i_k, _ ,_ ) = d_ids[did]
                                    toRemove.append(k_)
                                    toRemoveGt.append(i_k)                                    
                                    ious[k] = interDict[did]/unionDict[did]
                                    mask = (ious[k] > iouv[0]).cpu()
                                    pass                    
                                correct[pi[k]] = mask.cpu()  # iou_thres is 1xn
                                d_ids[did] = (k, i[k], interDict[did],unionDict[did])
                                # print("{} {} -- {}".format(k,i[k],ious_.shape))
                                pass
                            pass
                        pass
                    pass
                pass

            a = np.arange(len(correct))
            b = torch.tensor(toRemove).data.numpy()
            a = np.delete(a,b)
            
            # Append statistics (correct, conf, pcls, tcls,tids)
            stat = (correct, pred[:, 4].cpu(), pred[:, 5].cpu(),tcls,tids)
            #stat = (correct[a], pred[a, 4].cpu(), pred[a, 5].cpu(),
            #        np.array(tcls), np.array(tids))
            stats.append(stat)
            pass
        pass
    
    #for s in stats:
    #    (correct, conf, pcls, tcls,tids) = s
    
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # allLabelIds = [[np.concatenate(x, 0) for x in zip(*allLabelIds)] ]
    
    result = stats
    if len(stats):
        stat,plt = my_ap_per_class(*stats)
        p, r, ap, f1, ap_class = stat

        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
        pass
    '''
    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    if save_json and map and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except:
            print('WARNING: missing pycocotools package, can not compute official COCO mAP.\
            See requirements.txt.')
    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    '''
    return result

'''
print("=== 30 cm resolution ==\n\n")
stats = my_test(cfg='cfg/yolov3-spp.cfg',
                data='/data/zjc4/chipped-30/xview_data.txt',
                weights='weights/best-30.pt',
                batch_size=32,
                img_size=416,
                conf_thres=0.001,
                iou_thres=0.25,
                save_json=True)

print("=== 90 cm resolution ==\n\n")
stats = my_test(cfg='cfg/yolov3-spp.cfg',
                data='/data/zjc4/chipped-90/xview_data.txt',
                weights='weights/best-90.pt',
                batch_size=32,
                img_size=416,
                conf_thres=0.001,
                iou_thres=0.25,
                save_json=True)
'''

def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         model=None,
         dataloader=None):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.png'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, img_size, batch_size, rect=True, single_cls=opt.single_cls)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Plot images with bounding boxes
        f = 'test_batch%g.png' % batch_i  # filename
        if batch_i < 1 and not os.path.exists(f):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(imgs)  # inference and training outputs
            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)  # nms
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(d[5])],
                                  'bbox': [floatn(x, 3) for x in box[di]],
                                  'score': floatn(d[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices
                    
                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious,_,_= box_iou(pred[pi, :4], tbox[ti])
                        ious, i = ious.max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (img_size, img_size, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except:
            print('WARNING: missing pycocotools package, can not compute official COCO mAP. See requirements.txt.')

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images 
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps





if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    print(opt)

    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls)

    elif opt.task == 'benchmark':  # mAPs at 320-608 at conf 0.5 and 0.7
        y = []
        for i in [320, 416, 512, 608]:  # img-size
            for j in [0.5, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

    elif opt.task == 'study':  # Parameter study
        y = []
        x = np.arange(0.4, 0.9, 0.05)  # iou-thres
        for i in x:
            t = time.time()
            r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, opt.img_size, opt.conf_thres, i, opt.save_json)[0]
            y.append(r + (time.time() - t,))
        np.savetxt('study.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

        # Plot
        fig, ax = plt.subplots(3, 1, figsize=(6, 6))
        y = np.stack(y, 0)
        ax[0].plot(x, y[:, 2], marker='.', label='mAP@0.5')
        ax[0].set_ylabel('mAP')
        ax[1].plot(x, y[:, 3], marker='.', label='mAP@0.5:0.95')
        ax[1].set_ylabel('mAP')
        ax[2].plot(x, y[:, -1], marker='.', label='time')
        ax[2].set_ylabel('time (s)')
        for i in range(3):
            ax[i].legend()
            ax[i].set_xlabel('iou_thr')
        fig.tight_layout()
        plt.savefig('study.jpg', dpi=200)
