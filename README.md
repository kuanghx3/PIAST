# PIAST
pytorch code for PIAST

The dataset used in the article is at the link: https://github.com/IntelligentSystemsLab/ST-EVCDP

If you use our code, please cite the articles:
@article{KUANG2024123059,
title = {A physics-informed graph learning approach for citywide electric vehicle charging demand prediction and pricing},
journal = {Applied Energy},
volume = {363},
pages = {123059},
year = {2024},
issn = {0306-2619},
doi = {https://doi.org/10.1016/j.apenergy.2024.123059},
url = {https://www.sciencedirect.com/science/article/pii/S0306261924004422},
author = {Haoxuan Kuang and Haohao Qu and Kunxiang Deng and Jun Li},
keywords = {Electric vehicle charging, Spatio-temporal prediction, Physics informed neural network, Energy pricing},
abstract = {A growing number of electric vehicles (EVs) is putting pressure on smart charging services. As a foundation of informing drivers of vacant charging facilities and rationalizing pricing, an effective approach for predicting citywide spatio-temporal EV charging demand that incorporates pricing information is required. Although many deep learning models have been carried out with the orientation of improving prediction accuracy and achieved good results, there is still a lack of prediction models that are economically interpretable and can consider the spillover effect of price adjustments on EV charging demand. To fill the research gaps, we propose a learning approach for accurate EV charging demand prediction and reasonable pricing, named PIAST, which enables the integration of convolutional feature engineering, spatio-temporal dual attention mechanism and physics-informed neural network training. On a dataset containing 18,061 EV charging piles, we demonstrate the state-of-the-art performance of the proposed model. The results empirically showed that the proposed approach achieved an average improvement of 14.78% in accuracy, compared to other representative models. Moreover, the model can uncover the price elasticity of demand during training, making the model interpretable. Finally, price adjustment experiments are conducted to demonstrate the self-influence and spillover effects of price.}
}

and

@misc{qu2023physicsinformed,
	title={A physics-informed and attention-based graph learning approach for regional electric vehicle charging demand prediction}, 
	author={Haohao Qu and Haoxuan Kuang and Jun Li and Linlin You},
	year={2023},
	eprint={2309.05259},
	archivePrefix={arXiv},
	primaryClass={cs.LG}
}
