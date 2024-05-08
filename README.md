# CS598_FTL_Trans
## Main Files
*Project.ipynb* summarizes the results of our experiments, analysis and todos.
https://drive.google.com/drive/folders/1plvfq7C7iA1rEZCnVqzNF48tM0uSq1Tt?usp=drive_link is our main folder with checkpoints and results.
https://drive.google.com/file/d/19ERN9piQvd4khvqml5n2gVxqwLe2ERdT/view?usp=drive_link is our video presentation.

## Setups
### Dataset
We use [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/). We refer users to the link for requesting access. 

Since our experimented model including vanilla language model as well as hierachical models,
different dataset pre-processing are needed.
The datasets pre-processing script for language model (BERT, Clinical BERT) is included in the 
notebook (https://github.com/yitzhao/CS598_FTL_Trans/blob/main/Project.ipynb).
The dataset pre-processing for hierachical BERT and FTL-Trans can be found at https://github.com/yitzhao/CS598_FTL_Trans/blob/main/mimic_json.py and https://github.com/yitzhao/CS598_FTL_Trans/blob/main/preprocessing_FTL_Trans.ipynb respecively.


### Pretrained Weights
Pretrained Model
In our paper, we initialize the transformer layer with [ClinicalBERT](https://github.com/kexinhuang12345/clinicalBERT/tree/master). We refer user to the link for requesting pre-trained model. You can also use some other pre-trained models, like [BERT](https://github.com/huggingface/transformers).



## Experiments
Currently, we have done 5 experiments.

### Bert-based readmission prediction using discharge summary
After following the instruction in project.ipynb *Data Preprocessing* section to create the dataset, the model is trained within the *Bert-based readmission prediction using discharge summary* section and the evaluation result is within the *Bert-based readmission prediction using discharge summary* section. 

### ClinicalBert-based readmission prediction using discharge summary
After following the instruction in project.ipynb *Data Preprocessing* section to create the dataset, the model is trained within the *ClinicalBert-based readmission prediction using discharge summary* section and the evaluation result is within the *ClinicalBert-based readmission prediction using discharge summary* section. 

### Ftl-Trans based readmission prediction using All Notes
After following the instruction in preprocessing_FTL_Trans.ipynb to create the dataset, the model is trained and evaluated with command
```
!python3 run_clbert_ftlstm.py --data_dir ./DATA/readmission/ --train_data train.csv --val_data val.csv --test_data test.csv --log_path ./log_readmission.txt --bert_model ./pretraining/ --embed_mode all --task_name FTL-Trans_Prediction --max_seq_length 128 --train_batch_size 32 --eval_batch_size 1 --learning_rate 2e-5 --num_train_epochs 3 --warmup_proportion 0.1 --max_chunk_num 32 --seed 42 --gradient_accumulation_steps 1 --output_dir ./exp_FTL-Trans --save_model
```

### Ftl-Trans based (adjust time decay) readmission prediction using All Notes
After following the instruction in preprocessing_FTL_Trans.ipynb to create the dataset, the model is trained and evaluated with command
```
!python3 run_clbert_ftlstm_test.py --data_dir ./DATA/readmission/ --train_data train.csv --val_data val.csv --test_data test.csv --log_path ./log_readmission.txt --bert_model ./pretraining/ --embed_mode all --task_name FTL-Trans_Prediction_test --max_seq_length 128 --train_batch_size 32 --eval_batch_size 1 --learning_rate 2e-5 --num_train_epochs 3 --warmup_proportion 0.1 --max_chunk_num 32 --seed 42 --gradient_accumulation_steps 1 --output_dir ./exp_FTL-Trans_test --save_model
```

### LSTM based readmission prediction using All Notes
After following the instruction in preprocessing_FTL_Trans.ipynb to create the dataset, the model is trained and evaluated with command
```
!python3 run_clbert_lstm.py --data_dir ./DATA/readmission/ --train_data train.csv --val_data val.csv --test_data test.csv --log_path ./log_readmission.txt --bert_model ./pretraining/ --embed_mode all --task_name LSTM --max_seq_length 128 --train_batch_size 32 --eval_batch_size 1 --learning_rate 2e-5 --num_train_epochs 3 --warmup_proportion 0.1 --max_chunk_num 32 --seed 42 --gradient_accumulation_steps 1 --output_dir ./exp_LSTM --save_model
```

