# CS598_FTL_Trans
## Main Files
*Project_Draft.ipynb* summarizes the results of our experiments, analysis and todos.

## Setups
### Dataset
We use [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/). We refer users to the link for requesting access. 

File system expected:
```
data/
  test.csv
  train.csv
  val.csv
```

### Pretraining
Pretrained Model
In our paper, we initialize the transformer layer with [ClinicalBERT](https://github.com/kexinhuang12345/clinicalBERT/tree/master). We refer user to the link for requesting pre-trained model. You can also use some other pre-trained models, like [BERT](https://github.com/huggingface/transformers).

## Experiments
Currently, we have done 3 experiments.

### Bert-based readmission prediction using discharge summary
After following the instruction in project_draft.ipynb *DATA* section to create the dataset, the model is trained within the *Bert-based readmission prediction using discharge summary* section and the evaluation result is within the *Bert-based readmission prediction using discharge summary* section. 

### ClinicalBert-based readmission prediction using discharge summary
After following the instruction in project_draft.ipynb *DATA* section to create the dataset, the model is trained within the *ClinicalBert-based readmission prediction using discharge summary* section and the evaluation result is within the *ClinicalBert-based readmission prediction using discharge summary* section. 


### Ftl-Trans based readmission prediction using discharge summary
After following the instruction in preprocessing_FTL_Trans.ipynb to create the dataset, the model is trained and evaluated with command
```
!python3 run_clbert_ftlstm.py --data_dir ./DATA/readmission/ --train_data train.csv --val_data val.csv --test_data test.csv --log_path ./log_readmission.txt --bert_model ./pretraining/ --embed_mode all --task_name FTL-Trans_Prediction --max_seq_length 128 --train_batch_size 32 --eval_batch_size 1 --learning_rate 2e-5 --num_train_epochs 3 --warmup_proportion 0.1 --max_chunk_num 32 --seed 42 --gradient_accumulation_steps 1 --output_dir ./exp_FTL-Trans --save_model
```

## TODOs
1. We will conduct more experiments, incl. different datasets and different models to predict readmission.
2. We will also explore and fix a few issues that we have identified while training FTL-Trans.

