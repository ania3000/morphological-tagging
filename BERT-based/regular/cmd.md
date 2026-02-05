```
pip install -r requirements.txt
python ossetic_morph.py \
(--output_dir <ваша директория>) \
--hf_token <ваш токен> \
--model_name <имя модели> \
--num_train_epochs <число эпох>
```
Заданы на значения по умолчанию, можно поменять:
```
--learning_rate (default=5e-5)
--eval_steps (default = 200)
--save_steps (default = 200) 
--per_device_train_batch_size (default = 8)
--per_device_eval_batch_size (default = 8)
```
