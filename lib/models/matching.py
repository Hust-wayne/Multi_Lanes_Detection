import torch
import pdb

INFINITY = 987654.


def match_proposals_with_targets(model, proposals, targets, t_pos=15., t_neg=20.):
    # repeat proposals and targets to generate all combinations
    num_proposals = proposals.shape[0]
    num_targets = targets.shape[0]
    # pad proposals and target for the valid_offset_mask's trick
    proposals_pad = proposals.new_zeros(proposals.shape[0], proposals.shape[1] + 1)
    proposals_pad[:, :-1] = proposals
    proposals = proposals_pad
    targets_pad = targets.new_zeros(targets.shape[0], targets.shape[1] + 1)
    targets_pad[:, :-1] = targets
    targets = targets_pad

    proposals = torch.repeat_interleave(proposals, num_targets,
                                        dim=0)  # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b]

    targets = torch.cat(num_proposals * [targets])  # applying this 2 times on [c, d] gives [c, d, c, d]

    # get start and the intersection of offsets

    ############## multly lane
    targets_starts = targets[:, model.num_lane_type] * model.n_strips
    proposals_starts = proposals[:, model.num_lane_type] * model.n_strips
    starts = torch.max(targets_starts, proposals_starts).round().long()
    ends = (targets_starts + targets[:, model.num_lane_type+2] - 1).round().long()
    lengths = ends - starts + 1
    ends[lengths < 0] = starts[lengths < 0] - 1
    lengths[lengths < 0] = 0  # a negative number here means no intersection,
    valid_offsets_mask = targets.new_zeros(targets.shape)
    all_indices = torch.arange(valid_offsets_mask.shape[0], dtype=torch.long, device=targets.device)
    valid_offsets_mask[all_indices, model.num_lane_type + 3 + starts] = 1.
    valid_offsets_mask[all_indices, model.num_lane_type + 3 + ends + 1] -= 1.
    
    ##   put a -1 on the `end` index, giving [0, 1, 0, -1, 0]
    ##   if lenght is zero, the previous line would put a one where it shouldnt be.
    ##   this -=1 (instead of =-1) fixes this
    ##   the cumsum gives [0, 1, 1, 0, 0], the correct mask for the offsets
    valid_offsets_mask = valid_offsets_mask.cumsum(dim=1) != 0.
    invalid_offsets_mask = ~valid_offsets_mask

    # compute distances
    # this compares [ac, ad, bc, bd], i.e., all combinations
    distances = torch.abs((targets - proposals) * valid_offsets_mask.float()).sum(dim=1) / (lengths.float() + 1e-9
                                                                                            )  # avoid division by zero
    distances[lengths == 0] = INFINITY
    invalid_offsets_mask = invalid_offsets_mask.view(num_proposals, num_targets, invalid_offsets_mask.shape[1])
    distances = distances.view(num_proposals, num_targets)  # d[i,j] = distance from proposal i to target j

    positives = distances.min(dim=1)[0] < t_pos   # all arget lane  中对应满足条件的 proposal 中的一条   size : proposals lins, ->bool
    negatives = distances.min(dim=1)[0] > t_neg

    if positives.sum() == 0:
        target_positives_indices = torch.tensor([], device=positives.device, dtype=torch.long)
    else:
        target_positives_indices = distances[positives].argmin(dim=1)   #   proposal lane  对应的 target lane索引
    invalid_offsets_mask = invalid_offsets_mask[positives, target_positives_indices]

    return positives, invalid_offsets_mask[:, :-1], negatives, target_positives_indices
