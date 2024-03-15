import sys, os
sys.path.append(os.getcwd())
# sys.path.append("/home/wu/GatechResearch/Zixuan/PrisonerEscape")
from models.decoder import SingleGaussianDecoderStd, SingleGaussianDecoderStdParameter, MixtureDensityDecoder, MixtureDensityMLP, SelectiveMixtureDensityMLP
from models.encoders import EncoderRNN, ContrastiveEncoderRNN, ModifiedContrastiveEncoderRNN
from models.model import Model
from models.multi_head_mog import MixureDecoderMultiHead
from models.multi_head_mog import *
# from connected_filtering_prediction.model import MixtureInMiddle
import torch.nn as nn
# from models.gnn.gnn_post_lstm import GNNPostLSTM

def configure_decoder(conf, dim, num_heads, multi_head):
    number_gaussians = conf["number_gaussians"]
    # if multi_head:
    if multi_head:
        if conf["model_type"] == "vrnn" or conf["model_type"] == "variational_gnn":
            from models.variational_rnn_decoder import VariationalMixtureDecoderMultiHead
            decoder = VariationalMixtureDecoderMultiHead(
                input_dim=dim,
                num_heads=num_heads,
                output_dim=2,
                num_gaussians=number_gaussians,
                kl_loss_wt=conf["kl_loss_weight"]
            )
            return decoder

    if conf["model_type"] == "contrastive_vector":
        loss_type = conf["contrastive_loss"]
        decoder = ModifiedContrastiveMixureDecoderMultiHead(
            input_dim=dim,
            num_heads=num_heads,
            output_dim=2,
            num_gaussians=number_gaussians,
            loss_type=loss_type
        )

    elif conf["model_type"] == "contrastive_gnn" or conf["model_type"] == "simclr_gnn" or conf[
        "model_type"] == "cpc_gnn":
        loss_type = conf["contrastive_loss"]
        if not multi_head and num_heads > 1:
            decoder = Seq2SeqContrastiveAutoregressiveDecoder(input_dim=dim,
                                                              num_heads=num_heads,
                                                              output_dim=2,
                                                              num_gaussians=number_gaussians)

        else:
            decoder = ContrastiveGNNMixureDecoderMultiHead(
                input_dim=dim,
                num_heads=num_heads,
                output_dim=2,
                num_gaussians=number_gaussians,
                loss_type=loss_type
            )

        # else:
        #     decoder = MixureDecoderMultiHead(
        #         input_dim=dim,
        #         num_heads=num_heads,
        #         output_dim=2,
        #         num_gaussians=number_gaussians,
        #     )
    else:
        decoder = MixureDecoderMultiHead(
            input_dim=dim,
            num_heads=num_heads,
            output_dim=2,  # output dimension is always 2 since this is in the middle
            num_gaussians=number_gaussians,
        )

    return decoder

def configure_vgnn_model(conf, num_heads, total_input_dim, multi_head, concat_dim):
    """
    Concat dimension is what we concatenate to the final gnn pooled state that we feed into decoder
    Use variational RNN instead of regular LSTM
    """
    from models.gnn.variational_gnn_encoder import VariationalGNN
    # hidden_dim = conf["hidden_dim"]
    # input_dim = 3
    # hidden_dim = 8
    hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hideout_timestep_dim = 3
    encoder = VariationalGNN(total_input_dim, hidden_dim, gnn_hidden_dim,
                             use_last_k_detections=conf["use_last_k_detections"], use_last_two_detections=conf["use_last_two_detections"])
    # dim = gnn_hidden_dim + hideout_timestep_dim

    dim = gnn_hidden_dim + concat_dim

    if conf["use_last_k_detections"]:
        dim += 24  # k * 3 (t, x, y) for the detections

    if conf["use_last_two_detections"]:
        dim += 11  # k * 3 (t, x, y) for the detections

    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_vrnn(conf, num_heads, multi_head):
    from models.variational_rnn_encoder import VariationalRNNEncoder
    hidden_dim = conf["hidden_dim"]
    encoder = VariationalRNNEncoder(
        input_dim=conf["input_dim"],
        hidden_dim=hidden_dim,
        z_dim=conf["z_dim"],
        num_layers=1)
    decoder = configure_decoder(conf, hidden_dim + hidden_dim, num_heads, multi_head)

    model = Model(encoder, decoder)
    return model

def configure_gnn_model(conf, num_heads, total_input_dim, multi_head, concat_dim, last_k_fugitive_detection_bool, last_two_detection_with_vel_bool, start_location):
    from models.gnn.gnn import GNNLSTM
    # hidden_dim = conf["hidden_dim"]
    # input_dim = 3
    # hidden_dim = 8
    hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]

    encoder = GNNLSTM(total_input_dim, hidden_dim, gnn_hidden_dim, 
        last_k_fugitive_detection_bool= last_k_fugitive_detection_bool, 
        last_two_detection_with_vel_bool=last_two_detection_with_vel_bool,
        start_location = start_location)
    dim = gnn_hidden_dim + concat_dim
    # if conf["use_last_k_detections"]:
    #     dim += 24  # k * 3 (t, x, y) for the detections
    # if conf["use_last_two_detections"]:
    #     dim += 3
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model


def configure_simclr_gnn_model(conf, num_heads, total_input_dim, multi_head):
    from models.gnn.gnn_simclr import SimCLRGNN
    # hidden_dim = conf["hidden_dim"]
    # input_dim = 3
    # hidden_dim = 8
    hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hideout_timestep_dim = 3
    # hideout_timestep_location_dim = 5
    autoregressive = False
    if "autoregressive" in conf.keys():
        autoregressive = conf["autoregressive"]

    encoder = SimCLRGNN(total_input_dim, hidden_dim, gnn_hidden_dim,
                        use_last_k_detections=conf["use_last_k_detections"],
                        autoregressive=autoregressive)
    dim = gnn_hidden_dim + hideout_timestep_dim
    if conf["use_last_k_detections"]:
        dim += 24  # k * 3 (t, x, y) for the detections
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model


def configure_cpc_gnn_model(conf, num_heads, total_input_dim, multi_head):
    from models.gnn.gnn_cpc import CPCGNN
    # hidden_dim = conf["hidden_dim"]
    # input_dim = 3
    # hidden_dim = 8
    hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hideout_timestep_dim = 3
    # hideout_timestep_location_dim = 5
    autoregressive = False
    if "autoregressive" in conf.keys():
        autoregressive = conf["autoregressive"]

    encoder = CPCGNN(total_input_dim, hidden_dim, gnn_hidden_dim, use_last_k_detections=conf["use_last_k_detections"],
                     autoregressive=autoregressive)
    dim = gnn_hidden_dim + hideout_timestep_dim
    if conf["use_last_k_detections"]:
        dim += 24  # k * 3 (t, x, y) for the detections
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_mlp_model(conf, total_input_dim):
    model = MixtureDensityMLP(input_dim=total_input_dim, hidden=conf["hidden_dim"], output_dim=2, num_gaussians=conf["number_gaussians"], non_linear=nn.ReLU())
    return model

def configure_sel_mlp_model(conf, dynamic_input_dim, prior_network, sel_input_dim):
    model = SelectiveMixtureDensityMLP(input_dim=dynamic_input_dim, prior_network=prior_network, sel_dim=sel_input_dim, hidden=conf["hidden_dim"], output_dim=2, num_gaussians=conf["number_gaussians"], non_linear=nn.ReLU())
    return model

def configure_hetero_gnn_lstm_front_model(conf, num_heads, total_input_dim, multi_head, concat_dim, last_k_fugitive_detection_bool, start_location):
    from models.gnn.decoupled_hetero_lstm import LSTMHeteroPost 
    # hidden_dim = conf["hidden_dim"]
    # input_dim = 3
    # hidden_dim = 8
    hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    # hideout_timestep_dim = 3
    # hideout_timestep_location_dim = 5

    encoder = LSTMHeteroPost(total_input_dim, hidden_dim, gnn_hidden_dim, 1, start_location, last_k_fugitive_detection_bool)
    # dim = gnn_hidden_dim + hideout_timestep_dim
    dim = gnn_hidden_dim + concat_dim
    if start_location:
        dim += 2
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_contrastive_gnn_model(conf, num_heads, total_input_dim, multi_head):
    from models.gnn.gnn import ContrastiveGNNLSTM
    # hidden_dim = conf["hidden_dim"]
    # input_dim = 3
    # hidden_dim = 8
    hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hideout_timestep_dim = 3
    # hideout_timestep_location_dim = 5

    encoder = ContrastiveGNNLSTM(total_input_dim, hidden_dim, gnn_hidden_dim, use_last_k_detections=False)
    dim = gnn_hidden_dim + hideout_timestep_dim
    # if conf["use_last_k_detections"]:
    #     dim += 24  # k * 3 (t, x, y) for the detections
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model
    
def configure_gc_lstm_gnn(conf, num_heads, total_input_dim, multi_head):
    from models.gnn.gclstm import GCLSTMPrisoner
    gnn_hidden_dim = conf["gnn_hidden_dim"]

    hideout_timestep_dim = 3

    encoder = GCLSTMPrisoner(total_input_dim, gnn_hidden_dim, 1)
    dim = gnn_hidden_dim + hideout_timestep_dim
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_hybrid_gnn(conf, num_heads, total_input_dim, multi_head, last_k_fugitive_detection_bool, start_location):
    from models.gnn.hybrid_gnn import HybridGNN
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hidden_dim = conf["hidden_dim"]

    encoder = HybridGNN(total_input_dim, gnn_hidden_dim, hidden_dim, last_k_fugitive_detection_bool, start_location)
    if last_k_fugitive_detection_bool:
        hidden_dim += 24
    if start_location:
        hidden_dim += 2

    decoder = configure_decoder(conf, hidden_dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model


def configure_connected_model(conf, num_heads):
    encoder_type = conf["encoder_type"]
    hidden_dim = conf["hidden_dim"]
    decoder_type = conf["decoder_type"]
    number_gaussians = conf["number_gaussians"]
    input_dim = conf["input_dim"]
    hidden_connected_bool = conf["hidden_connected"]

    if encoder_type == "lstm":
        encoder = EncoderRNN(input_dim, hidden_dim)
    mixture_decoder = MixtureDensityDecoder(
        input_dim=hidden_dim,
        output_dim=2,  # output dimension is always 2 since this is in the middle
        num_gaussians=number_gaussians,
    )

    if hidden_connected_bool:
        decoder_input_dim = 2 + hidden_dim
    else:
        decoder_input_dim = 2

    # location_decoder = SingleGaussianDecoderStd(2, output_dim=2*num_heads) # could switch to mixture as well
    if decoder_type == "single_gaussian":
        # We multiply the number of heads by 2 for each dimension in (x, y)
        location_decoder = SingleGaussianDecoderStd(decoder_input_dim, output_dim=2 * num_heads)
    elif decoder_type == "mixture":
        location_decoder = MixureDecoderMultiHead(
            input_dim=hidden_dim,
            num_heads=num_heads,
            output_dim=2,
            num_gaussians=number_gaussians,
        )

    model = MixtureInMiddle(encoder, mixture_decoder, location_decoder, hidden_connected_bool)
    return model


def configure_regular(conf, num_heads, multi_head):
    """_summary_

    Args:
        conf (dict): The model configuration dictionary from the yaml config file. 
        num_heads (int): Represents the number of heads of the model.

    Returns:
        _type_: pytorch model
    """
    encoder_type = conf["encoder_type"]
    hidden_dim = conf["hidden_dim"]
    decoder_type = conf["decoder_type"]
    number_gaussians = conf["number_gaussians"]
    input_dim = conf["input_dim"]

    if encoder_type == "lstm":
        encoder = EncoderRNN(input_dim, hidden_dim)

    decoder = configure_decoder(conf, hidden_dim, num_heads, multi_head)
    # if decoder_type == "single_gaussian":
    #     # We multiply the number of heads by 2 for each dimension in (x, y)
    #     decoder = SingleGaussianDecoderStd(hidden_dim, output_dim=2*num_heads)
    # elif decoder_type == "mixture":
    #     decoder = MixureDecoderMultiHead(
    #         input_dim=hidden_dim,
    #         num_heads = num_heads,
    #         output_dim=2,
    #         num_gaussians=number_gaussians,
    #     )
    #     # decoder = MixtureDensityDecoder(
    #     #     input_dim=hidden_dim,
    #     #     # input_dim = 16,
    #     #     output_dim=2, # output dimension is always 2 since this is in the middle
    #     #     num_gaussians=number_gaussians,
    #     # )

    print(decoder_type)

    model = Model(encoder, decoder)

    return model


def configure_contrastive_vector(conf, num_heads, multi_head):
    """_summary_

    Args:
        conf (dict): The model configuration dictionary from the yaml config file.
        num_heads (int): Represents the number of heads of the model.

    Returns:
        _type_: pytorch model
    """
    encoder_type = conf["encoder_type"]
    hidden_dim = conf["hidden_dim"]
    decoder_type = conf["decoder_type"]
    number_gaussians = conf["number_gaussians"]
    input_dim = conf["input_dim"]

    # if encoder_type == "contrastive_lstm":
    encoder = ModifiedContrastiveEncoderRNN(input_dim, hidden_dim)
    # else:
    #     encoder = EncoderRNN(input_dim, hidden_dim)

    decoder = configure_decoder(conf, hidden_dim, num_heads, multi_head)
    print(decoder_type)

    model = Model(encoder, decoder)

    return model


def configure_gnn_lstm_model(conf):
    # hidden_dim = conf["hidden_dim"]
    input_dim = 3
    hidden_dim = 8
    hideout_timestep_dim = 3

    number_gaussians = conf["number_gaussians"]

    encoder = GNNPostLSTM(input_dim, hidden_dim)
    decoder = MixtureDensityDecoder(
        input_dim=hidden_dim + hideout_timestep_dim,
        # input_dim = 16,
        output_dim=2,  # output dimension is always 2 since this is in the middle
        num_gaussians=number_gaussians,
    )

    model = Model(encoder, decoder)
    return model


def configure_stn(conf, num_heads, total_input_dim, multi_head):
    from models.gnn.stn import STGCNEncoder
    # hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hideout_timestep_dim = 3
    periods = 16
    batch_size = 128

    encoder = STGCNEncoder(total_input_dim, gnn_hidden_dim, periods, batch_size)
    dim = gnn_hidden_dim + hideout_timestep_dim
    if conf["use_last_k_detections"]:
        dim += 24  # k * 3 (t, x, y) for the detections
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model


def configure_temporal_gcn(conf, num_heads, total_input_dim, multi_head):
    from experimental.temporal_gcn import TemporalGNN
    # hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hideout_timestep_dim = 3
    periods = 16
    batch_size = 128

    encoder = TemporalGNN(node_features=total_input_dim,
                          hidden_dim=gnn_hidden_dim,
                          periods=periods,
                          batch_size=batch_size,
                          use_last_k_detections=conf["use_last_k_detections"])

    dim = gnn_hidden_dim + hideout_timestep_dim
    if conf["use_last_k_detections"]:
        dim += 24  # k * 3 (t, x, y) for the detections
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model


def configure_hetero_lstm(conf, num_heads, total_input_dim, multi_head):
    from models.gnn.hetero_gc_lstm_batch import HeteroLSTM
    # hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]

    encoder = HeteroLSTM(gnn_hidden_dim)
    decoder = configure_decoder(conf, gnn_hidden_dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_cvae(conf, num_heads, multi_head):
    from models.cvae.model_continuous import CVAEContinuous
    gmm_bool = conf["gmm_bool"]
    encoder_hidden_dim = conf["encoder_hidden_dim"]
    future_hidden_dim = conf["future_hidden_dim"]
    z_dim = conf["z_dim"]
    x_dim = conf["input_dim"]
    # decoder_input_dim = z_dim * 2
    # decoder = configure_decoder(conf, decoder_input_dim, num_heads, multi_head)
    cvae = CVAEContinuous(x_dim, encoder_hidden_dim, future_hidden_dim, z_dim, gmm_decoder=gmm_bool)

    return cvae

def configure_gmm_cvae(conf, num_heads, multi_head):
    from models.cvae.model_mixture import CVAEContinuous
    encoder_hidden_dim = conf["encoder_hidden_dim"]
    future_hidden_dim = conf["future_hidden_dim"]
    x_dim = conf["input_dim"]
    # decoder_input_dim = z_dim * 2
    # decoder = configure_decoder(conf, decoder_input_dim, num_heads, multi_head)
    cvae = CVAEContinuous(x_dim, encoder_hidden_dim, future_hidden_dim)

    return cvae

def configure_gmm_cvae_z(conf, num_heads, multi_head):
    from models.cvae.model_mixture import CVAEMixture
    decoder_type = conf["decoder_type"]
    z_dim = conf["z_dim"]
    encoder_hidden_dim = conf["encoder_hidden_dim"]
    future_hidden_dim = conf["future_hidden_dim"]
    z_dim = conf["z_dim"]
    x_dim = conf["input_dim"]
    # decoder_input_dim = z_dim * 2
    # decoder = configure_decoder(conf, decoder_input_dim, num_heads, multi_head)
    cvae = CVAEMixture(x_dim, encoder_hidden_dim, future_hidden_dim, z_dim, num_heads, decoder_type=decoder_type)

    return cvae

def configure_gmm_gnn_cvae(conf, num_heads, multi_head, total_input_dim):
    from models.cvae.model_continuous import CVAEContinuous
    encoder_hidden_dim = conf["encoder_hidden_dim"]
    future_hidden_dim = conf["future_hidden_dim"]
    z_dim = conf["z_dim"]
    gmm_decoder = conf["gmm_bool"]

    model = CVAEContinuous(total_input_dim, encoder_hidden_dim, future_hidden_dim, z_dim, gmm_decoder=gmm_decoder, input_type = "gnn")
    return model

def configure_vector_cnn(conf, num_heads, multi_head):
    from models.cnn.resnet import ResNet50
    from models.encoder_cnn_lstm import Encoder

    hidden_dim = conf["hidden_dim"]
    input_dim = conf["input_dim"]
    blue_obs_encoder = EncoderRNN(input_dim, hidden_dim)

    out_dim = 128
    h = 64
    res_net = ResNet50(out_dim=out_dim, channels=1)
    encoder = Encoder(blue_obs_encoder, res_net, hidden_dim, out_dim, h)

    decoder = configure_decoder(conf, h, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_mog_v2(conf):
    from models.mi.mi_blue import BlueMIMixture
    input_dim = conf["input_dim"]
    h1 = conf["h1"]
    h2 = conf["h2"]
    output_dim = 2
    number_gaussians = conf["number_gaussians"]
    return BlueMIMixture(input_dim, output_dim, num_mixtures=number_gaussians, h1=h1, h2=h2)

def get_input_dim(config):
    total_input_dim = 3
    if config["datasets"]["last_two_detection_in_graph_obs"]: 
        total_input_dim += 6
    if config["datasets"]["one_hot_agents"]:
        total_input_dim += 3
    if config["datasets"]["detected_location"]:
        total_input_dim += 2
    if config["datasets"]["timestep"]:
        total_input_dim += 1
    return total_input_dim

def configure_model(config, pri_or_combine=None, prior_network=None):

    if pri_or_combine is not None:
        conf = config[pri_or_combine]
    else:
        conf = config["model"]

    if conf["model_type"] == "padded_mog_v2":
        return configure_mog_v2(conf)


    num_heads = config["datasets"]["num_heads"]
    if config["datasets"]["include_current"]:
        num_heads += 1

    multi_head = config["datasets"]["multi_head"]
    last_k_fugitive_detection_bool = config["datasets"]["get_last_k_detections"]
    last_two_fugitive_detection_bool = config["datasets"]["last_two_detection_with_vel"]
    start_location = config["datasets"]["get_start_location"]

    if conf["model_type"] == "connected":
        print("connected model")
        return configure_connected_model(conf, num_heads)
    elif conf["model_type"] == "gnn":
        total_input_dim = get_input_dim(config)
        concat_dim = 3
        if config["datasets"]["get_start_location"]:
            concat_dim += 2
        # if config["datasets"]["get_last_k_detections"]:
        #     concat_dim += 24
        return configure_gnn_model(conf, num_heads, total_input_dim, multi_head, concat_dim, last_k_fugitive_detection_bool, last_two_fugitive_detection_bool, start_location)
    elif conf["model_type"] == "mlp":
        total_input_dim = 3
        return configure_mlp_model(conf, total_input_dim)
    elif conf["model_type"] == "sel_mlp":
        dynamic_input_dim = 3
        sel_input_dim = 6
        return configure_sel_mlp_model(conf, dynamic_input_dim, prior_network, sel_input_dim)
    elif conf["model_type"] == "hetero_gnn_lstm_front":
        total_input_dim = get_input_dim(config)
        concat_dim = 3
        if config["datasets"]["get_start_location"]:
            concat_dim += 2
        start_location_bool = config["datasets"]["get_start_location"]
        return configure_hetero_gnn_lstm_front_model(conf, num_heads, total_input_dim, multi_head, concat_dim, last_k_fugitive_detection_bool, start_location_bool)
    elif conf["model_type"] == "simclr_gnn":
        total_input_dim = get_input_dim(config)
        return configure_simclr_gnn_model(conf, num_heads, total_input_dim, multi_head)
    elif conf["model_type"] == "cpc_gnn":
        total_input_dim = get_input_dim(config)
        return configure_cpc_gnn_model(conf, num_heads, total_input_dim, multi_head)
    elif conf["model_type"] == "contrastive_gnn":
        total_input_dim = get_input_dim(config)
        return configure_contrastive_gnn_model(conf, num_heads, total_input_dim, multi_head)

    elif conf["model_type"] == "stn":
        total_input_dim = get_input_dim(config)
        return configure_stn(conf, num_heads, total_input_dim, multi_head)
    elif conf["model_type"] == "temp_gcn":
        total_input_dim = get_input_dim(config)
        return configure_temporal_gcn(conf, num_heads, total_input_dim, multi_head)
    elif conf["model_type"] == "hetero_lstm":
        total_input_dim = get_input_dim(config)
        return configure_hetero_lstm(conf, num_heads, total_input_dim, multi_head)
    elif conf["model_type"] == "cvae":
        return configure_cvae(conf, num_heads, multi_head)
    elif conf["model_type"] == "gmm_cvae":
        return configure_gmm_cvae(conf, num_heads, multi_head)
    elif conf["model_type"] == "gmm_cvae_z":
        return configure_gmm_cvae_z(conf, num_heads, multi_head)
    elif conf["model_type"] == "gmm_cvae_gnn":
        total_input_dim = get_input_dim(config)
        return configure_gmm_gnn_cvae(conf, num_heads, multi_head, total_input_dim)
    elif conf["model_type"] == "hybrid_gnn":
        total_input_dim = get_input_dim(config)
        return configure_hybrid_gnn(conf, num_heads, total_input_dim, multi_head, last_k_fugitive_detection_bool, start_location)
    elif conf["model_type"] == "vector_cnn":
        return configure_vector_cnn(conf, num_heads, multi_head)
    elif conf["model_type"] == "variational_gnn":
        total_input_dim = get_input_dim(config)
        if config["datasets"]["get_start_location"]:
            concat_dim = 5
        else:
            concat_dim = 3
        return configure_vgnn_model(conf, num_heads, total_input_dim, multi_head, concat_dim)
    elif conf["model_type"] == "contrastive_vector":
        return configure_contrastive_vector(conf, num_heads, multi_head)
    elif conf["model_type"] == "vrnn":
        return configure_vrnn(conf, num_heads, multi_head)
    else:
        return configure_regular(conf, num_heads, multi_head)
