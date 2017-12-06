import torch
import torch.nn as nn
import logging
logger = logging.getLogger("dev")


class NCELoss(nn.Module):
    def __init__(self):
        super(NCELoss, self).__init__()

    """
    loss = log sig(c't) + E[log sig(-c'n)]
    """
    def forward(self, context_embeddings, target_embeddings, noise_embeddings):
        # contexts (batch_size * dim)
        # targets (batch_size * dim)
        # noise (num_noise * dim)

        context_target_element_wise = context_embeddings * target_embeddings
        # -> batch_size * dim
        logger.warning(["context_target: ", context_target_element_wise.size()])

        context_target_cross_products = torch.sum(context_target_element_wise, dim=1)
        # -> batch_size * 1
        logger.warning(["context_target_cross_products: ", context_target_cross_products.size()])

        context_noise_cross_products = torch.mm(context_embeddings, torch.transpose(noise_embeddings,0,1))
        # -> batch_size * num_noise

        context_target_sigmoid = torch.sigmoid(context_target_cross_products)
        context_noise_sigmoid = torch.sigmoid(-context_noise_cross_products)

        log_context_target_sigmoid = torch.log(context_target_sigmoid)
        log_context_noise_sigmoid = torch.log(context_noise_sigmoid)
        context_noise_average_log_sigmoid = torch.mean(log_context_noise_sigmoid,dim=1)

        loss_vector = torch.add(log_context_target_sigmoid,context_noise_average_log_sigmoid)
        logger.warning(["loss_vector: ", loss_vector.size()])
        # -> batch_size * 1

        return torch.mean(loss_vector)








