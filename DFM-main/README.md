# CFMRec
# CFMRec: Cascaded Flow Matching for Socially-aware Sequential Recommendation
This is a Tensorflow implementation for DFM.
* Bin WU, Xinxin WU, Haokai MA, Yinwei Wei, Mingliang XU, and Tat-Seng CHUA
# Environment Requirement
The code has been tested running under Python 3.7.12. The required packages are as follows:
- tensorflow-gpu==2.10.0
- numpy==1.21.6
- scipy==1.7.3
- pandas==1.3.5
# Dataset
We provide Gowalla_yiding_u5_s3(Gowalla) dataset.
- .\dataset\Gowalla_yiding_u5_s3.rating and Gowalla_yiding_u5_s3.uu
- Each line is a user with her/his positive interactions with items: userID \ itemID \ ratings \time.
- Each user has more than 10 associated actions.
# run CFMRec:
run main.py in IDE or with command line:
```bash
python main.py
```



