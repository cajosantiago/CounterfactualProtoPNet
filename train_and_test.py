import time
import torch

from helpers import list_of_distances, make_one_hot

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total_mask_distance_diference = 0
    n_masks = 0
    total_activation = 0
    total_masked_activation = 0
    total_positive_activation = 0
    total_positive_masked_activation = 0
    total_negative_activation = 0
    total_negative_masked_activation = 0
    total_activation = 0
    total_masked_activation = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_fine_cost = 0

    for i, (image, label, mask) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()
        input2 = mask.cuda()
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances, masked_output, min_masked_distances,activations,activation,masked_activation = model(input,input2)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output,target)

            masked_output = masked_output[torch.count_nonzero(input2,dim=(2,3)).reshape((label.shape[0],)) != 0]
            min_masked_distances = min_masked_distances[torch.count_nonzero(input2,dim=(2,3)).reshape((label.shape[0],)) != 0]
            label2 = torch.zeros(masked_output.shape[0],dtype=torch.int64)
            target2 = label2.cuda()
            masked_cross_entropy = torch.nn.functional.cross_entropy(masked_output,target2)
            cross_entropy = torch.nn.functional.cross_entropy(torch.cat((output,masked_output)),torch.cat((target,target2)))

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)
                
                masked_prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label2]).cuda()
                inverted_masked_distances, _ = torch.max((max_dist - min_masked_distances) * masked_prototypes_of_correct_class, dim=1)
                masked_cluster_cost = torch.mean(max_dist - inverted_masked_distances)
                cluster_cost = torch.mean(max_dist - torch.cat((inverted_distances,inverted_masked_distances)))

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                masked_prototypes_of_wrong_class = 1 - masked_prototypes_of_correct_class
                inverted_masked_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_masked_distances) * masked_prototypes_of_wrong_class, dim=1)
                masked_separation_cost = torch.mean(max_dist - inverted_masked_distances_to_nontarget_prototypes)
                separation_cost = torch.mean(max_dist - torch.cat((inverted_distances_to_nontarget_prototypes,inverted_masked_distances_to_nontarget_prototypes)))

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

                fine_annotation_cost = 0
                proto_num_per_class = model.module.num_prototypes // model.module.num_classes
                all_white_mask = torch.ones(mask.shape[2], mask.shape[3]).cuda()
                bool_mask = input2==1
                for index in range(image.shape[0]):
                    if torch.sum(bool_mask[index]).item() != 0:
                        fine_annotation_cost += torch.norm(activations[index, :label[index] * proto_num_per_class] * (1 * all_white_mask)) + \
                            torch.norm(activations[index, label[index] * proto_num_per_class : (label[index] + 1) * proto_num_per_class] * (1 * bool_mask[index])) + \
                                torch.norm(activations[index, (label[index]+1) * proto_num_per_class:] * (1 * all_white_mask))

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            tp += (torch.logical_and(predicted==target ,predicted==1)).sum().item()
            fp += (torch.logical_and(predicted!=target ,predicted==1)).sum().item()
            tn += (torch.logical_and(predicted==target ,predicted==0)).sum().item()
            fn += (torch.logical_and(predicted!=target ,predicted==0)).sum().item()
            total_mask_distance_diference += torch.sum(min_masked_distances-min_distances[torch.count_nonzero(input2,dim=(2,3)).reshape((label.shape[0],)) != 0]).item()
            n_masks += masked_output.shape[0]


            total_activation += activation[torch.count_nonzero(input2,dim=(2,3)).reshape((label.shape[0],)) != 0].sum().item()
            total_masked_activation += masked_activation[torch.count_nonzero(input2,dim=(2,3)).reshape((label.shape[0],)) != 0].sum().item()

            negative_activation,positive_activation = torch.chunk(activation,2,dim=1)
            negative_masked_activation,positive_masked_activation = torch.chunk(masked_activation,2,dim=1)
            total_positive_activation += positive_activation[torch.count_nonzero(input2,dim=(2,3)).reshape((label.shape[0],)) != 0].sum().item()
            total_positive_masked_activation += positive_masked_activation[torch.count_nonzero(input2,dim=(2,3)).reshape((label.shape[0],)) != 0].sum().item()
            total_negative_activation += negative_activation[torch.count_nonzero(input2,dim=(2,3)).reshape((label.shape[0],)) != 0].sum().item()
            total_negative_masked_activation += negative_masked_activation[torch.count_nonzero(input2,dim=(2,3)).reshape((label.shape[0],)) != 0].sum().item()
            
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            if fine_annotation_cost != 0:
                total_fine_cost += fine_annotation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1
                          #+ coefs['msk_crs_ent'] * masked_cross_entropy
                          #+ coefs['msk_clst'] * masked_cluster_cost
                          #+ coefs['msk_sep'] * masked_separation_cost
                          #+ coefs['fine'] * fine_annotation_cost
                          )
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del input2
        del target
        del target2
        del output
        del predicted
        del min_distances
        del min_masked_distances
        del activation
        del masked_activation

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\tfine: \t{0}'.format(total_fine_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
    log('\ttp: \t{0}'.format(tp))
    log('\tfp: \t{0}'.format(fp))
    log('\ttn: \t{0}'.format(tn))
    log('\tfn: \t{0}'.format(fn))
    log('\ttotal mask distance diff:\t{0}'.format(total_mask_distance_diference))
    log('\ttotal mask activation diff:\t{0}'.format(total_activation-total_masked_activation))
    log('\tratio of activation in masked area:\t{0}'.format((total_activation-total_masked_activation) / total_activation))
    log('\tratio of negative activation in masked area:\t{0}'.format((total_negative_activation-total_negative_masked_activation) / total_negative_activation))
    log('\tratio of positive activation in masked area:\t{0}'.format((total_positive_activation-total_positive_masked_activation) / total_positive_activation))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
