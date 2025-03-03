import os

from ibydmt import Config, register_config


@register_config(name="vindr")
class CheXpertConfig(Config):
    def __init__(self):
        super().__init__()
        self.name = os.path.basename(__file__).replace(".py", "")

        data = self.data
        data.dataset = "vindr"
        # data.backbone = "cxr-bert-specialized"
        # data.backbone = "chexpert-resnet"
        # data.bottleneck = "chexpert_attribute"
        # data.classifier = "zeroshot"
        # data.sampler = "attribute"
        # data.num_concepts = 12

        # testing = self.testing
        # testing.significance_level = 0.05
        # testing.wealth = "ons"
        # testing.bet = "tanh"
        # testing.kernel = "rbf"
        # testing.kernel_scale_method = "quantile"
        # testing.kernel_scale = [0.3, 0.4, 0.5, 0.6, 0.7]
        # testing.tau_max = [200, 400]
        # testing.images_per_class = 20
        # testing.cardinality = [1, 2, 4]
        # testing.r = 20
