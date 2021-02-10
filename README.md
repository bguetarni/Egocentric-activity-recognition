The report is available at: https://github.com/bguetarni/ST50

## Literature

Damen, D., Doughty, H., Farinella, G. M., Furnari, A.,Kazakos, E., Ma, J., Moltisanti, D., Munro, J., Perrett, T., Price, W., et al.(2020). Rescaling egocentric vision. [arxiv](https://arxiv.org/pdf/2006.13256.pdf)

Sudhakaran, S., Escalera, S., and Lanz, O. (2019). LSTA: Long short-term attention for egocentric action recognition. [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sudhakaran_LSTA_Long_Short-Term_Attention_for_Egocentric_Action_Recognition_CVPR_2019_paper.pdf)

Ye, W., Cheng, J., Yang, F., and Xu, Y. (2019). Two-streamconvolutional network for improving activity recognition using convolu-tional long short-term memory networks. [ieee](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8721710)

Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K.,and Woo, W.-c. (2015). Convolutional LSTM network: A machine learningapproach for precipitation nowcasting. [arxiv](https://arxiv.org/pdf/1506.04214.pdf)

Wang, L., Xiong, Y., Wang, Z., Qiao, Y., Lin, D., Tang,X., and Van Gool, L. (2016). Temporal segment networks: Towards goodpractices for deep action recognition. [arxiv](https://arxiv.org/pdf/1608.00859.pdf)

## Data
To download the data (training, validation, test) see https://github.com/epic-kitchens/epic-kitchens-download-scripts

The script ```epic_downloader.py``` is already provided in this repo. You will have to extract manually the data from the downladed archive before using it.

Annotations (ground-truth) are available at https://github.com/epic-kitchens/epic-kitchens-100-annotations

[EPIC-KITCHENS 100 website](https://epic-kitchens.github.io/2020-100.html)

[EPIC-KITCHENS 55 website](https://epic-kitchens.github.io/2020-55.html)

## Training
To train a model use the script ```train.py```.

## Testing
To evaluate a model use the script ```test.py```. As the test set labels are not available now, you should use the validation set for evaluation.

**Before training or testing a model, be sure that the data is saved as numpy arrays.**
**The script ```data.py``` provides a routine for that.**

## Connexion to visdom

To visualize the plots during the training, you must use visdom; and to do so creating the session beforehand with:
```
python -m visdom.server -p XXXX
```
where ```XXXX``` is any available port. It will use localhost as server. To visualize the curves open ```localhost:XXXX``` in a browser.

To use visdom with a remote server: [github-gist](https://gist.github.com/amoudgl/011ed6273547c9312d4f834416ab1d0c)  
