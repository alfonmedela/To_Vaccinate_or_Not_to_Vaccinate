# To Vaccinate or Not to Vaccinate: : It’s not a Question challenge!

**#ZindiWeekendz** hackathon **#top5** solution

## How to replicate solution

Libraries needed:
- Numpy
- Pandas
- Torch
- Fastai
- Transformers

Just in case you are not able to run it, I include a requirements.txt with all the dependencies I have installed in the conda environment. You can check the versions there.

### Running the algorithm
Repeat step 1 and 2 for every split (0,1,2,3,4):
1) Run *split_X/main_train.py*
2) Run *split_X/submission.py*

Once all submission files are generated, run *final_submission.py*

## Method
I used Roberta model with a final layer of a single neuron to predict a value between -1 and 1. The network is trained with MSE loss and evaluated with RMSE metric as it is the objective metric in the competition. I used AdamW optimizer and trained two different stages, first all freezed but the final regression layer and finally, the whole network. 

As the dataset is really small, it is 10,000 training samples, each split obtained different results in the public leaderboard. Therefore, I decided to stack 4 models trained on different splits. Additionally, I trained a last 5th split dropping the samples with 0.33 agreement. This way, the network learned from samples with higher confidence labels. 



