# Revisiting the Tag Relevance Prediction Problem

This work is licensed under the Creative Commons Attribution-NonCommercial 3.0 License. If you are using these code and/or datasets, please cite the following papers:

- [Kotkov D., Maslov A., Neovius M. Revisiting the Tag Relevance Prediction Problem. In Proceedings of The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval](https://www.researchgate.net/publication/351347450_Revisiting_the_Tag_Relevance_Prediction_Problem)
- [Vig, J., Sen, S., & Riedl, J. (2012). The tag genome: Encoding community knowledge to support novel interaction. ACM Transactions on Interactive Intelligent Systems (TiiS), 2(3), 1-44.](https://dl.acm.org/doi/abs/10.1145/2362394.2362395)

## Dataset and code

- The preprocessed dataset (features) is available in the data folder
- The full dataset (raw data) will become available soon
- The baseline algorithm (Vig et al.) is available in the `r` folder
- The algorithm, which generates features will become available soon


## Hot to run


- R-script (Vig et al.): `./r/my_run_tenfolds.R`
- Python (TagDL): `train_pytorch.py`
- All data, train and test, predictions from the experiments reported in (Kotkov et al.), 10 folds are in `./data`
- Copied predictions for 10 folds: `cp .temp/ -> /data/10_folds_predictions_sigir2021`
- Backup data to archive `data_sigir_2021.tar.gz`
