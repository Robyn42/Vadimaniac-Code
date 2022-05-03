## Title: Argoverse’d

### Group Members

Robyn Logan - rlogan2

Hardy Bright - hbright

Shane Parr - sparr1

Ye Wang - ywang683

### Introduction

> What problem are you trying to solve and why? If you are doing something new, detail how you arrived at this topic and what motivated you. What kind of problem is this? Classiﬁcation? Regression? Structured prediction? Reinforcement Learning? Unsupervised Learning? Etc.

This is a time-series regression task. We are training and assessing the robustness of several kinds of models on the Argoverse 2 motion prediction dataset. We’re specifically interested in adversarial attacks on models which may be used in real self-driving cars, due to the serious ethical and technical implications that may follow from the brittleness of such models.

### Related Work

> Are you aware of any, or is there any prior work that you drew on to do your project? Please read and brieﬂy summarize (no more than one paragraph) at least one paper/article/blog relevant to your topic beyond the paper you are re-implementing/novel idea you are researching. In this section, also include URLs to any public implementations you ﬁnd of the paper you’re trying to implement. Please keep this as a “living list”--if you stumble across a new implementation later down the line, add it to this list.1

1. Argoverse 2 Dataset: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/4734ba6f3de83d861c3176a6273cac6d-Paper-round2.pdf Main benefits of this paper: A description of the dataset Table 4 provides a summary for the baseline models

2. Multimodal Motion Prediction with Stacked Transformers Paper: https://arxiv.org/pdf/2103.11624.pdf Github: https://github.com/decisionforce/mmTransformer A recent paper using stacked transformers to tackle the different modalities in the motion forecasting task. It claims to achieve comparable accuracy to LSTMs and includes ablation studies on whether or not social features (interactions between agents) or map features are used.

3. https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Fooling_LiDAR_Perception_via_Adversarial_Trajectory_Perturbation_ICCV_2021_paper.pdf A recent paper on adversarially attacking LiDAR based vehicle perception (this would be upstream of any kind of motion prediction)

4. What-If Motion Prediction for Autonomous Driving (current SOTA on Argoverse 1 motion prediction) Paper: https://arxiv.org/pdf/2008.10587.pdf Github: https://github.com/wqi/WIMP From the abstract: WIMP is “a recurrent graph-based attentional approach with interpretable geometric (actor-lane) and social (actor-actor) relationships that supports the injection of counterfactual geometric goals and social contexts.”

### Data

#### WHAT DATA ARE YOU USING (IF ANY)?

> If you’re using a standard dataset (e.g. MNIST), you can just mention that brieﬂy. Otherwise, say something more about where your data come from (especially if there’s anything interesting about how you will gather it).

Argoverse1 and Argoverse2. Argoverse2 is a new dataset released in 2021. Its API is under active development now.

#### HOW BIG IS IT?

> Will you need to do signiﬁcant preprocessing?

5GB, 58 GB. Preprocessing is significant and may take N^2 time.

### Methodology

#### WHAT IS THE ARCHITECTURE OF YOUR MODEL?

We currently have GRU and Multi-layer Perceptron models designed against Argoverse 2 data. As we move forward we will likely explore additional architectures including LSTM, Transformers and potentially hybrid setups that bring together the benefits of multiple setups.

#### HOW ARE YOU TRAINING THE MODEL?

Our current models are being trained using portions of the Argoverse 2 dataset. We are simultaneously exploring the structure of the Argoverse 1 dataset. The authors of both datasets have portioned off train, validation and test data but at this time we have used a smaller selection of this to develop ideas for handling data organization, preprocessing, model programming and performance. The data is multivariate where parallel streams of features have real values at each timestep. Our preprocessing exposes the time-series, tabular data within the parquet file format. It combines the continuous variable features per timestep from multiple files. The Multi-layer Perceptron is trained by conforming the data into N-Grams where each set of features within the timeste is a “word”. Currently, we are using an “5-gram” arrangement where 4 timesteps are the input with the 5th forming the “true value”. That model has been run with mean-squared error as the loss and for upwards of 10,000 epochs. The GRU is being trained with windows of timesteps where the inputs and “true values” are offset from each other by deleting beginning and ending timesteps.

As we develop more ideas for design and evaluation we will also continue to explore the Argoverse 1 dataset and its particular design needs.

#### IF YOU ARE DOING SOMETHING NEW, JUSTIFY YOUR DESIGN.

> Also note some backup ideas you may have to experiment with if you run into issues.

Given that our data is composed of parallel multivariate time series, the GRU architecture was chosen for its ability to handle sequence information.

The Multi-layer Perceptron can also find success with this type of data due to the inherent parallel nature of its dense layers.

As part of preprocessing, the dataset is new and no feature extraction code exists. We will extract social features, including computing the velocity of each agent in each scenario.

We will explore adversarial attacks of various forms, likely starting with a black-box attack, and then branching out as necessary. This is conditional on getting motion prediction working, however. One fall-back we have is to use the Argoverse 1 dataset, as it’s been around for longer, and so there are more existing implementations and documentation.

### Metrics

#### WHAT CONSTITUTES “SUCCESS?”

> What experiments do you plan to run? For most of our assignments, we have looked at the accuracy of the model. Does the notion of “accuracy” apply for your project, or is some other metric more appropriate?

Our project takes into account multiple measures of success. One being the accuracy of the motion forecasting predictions with regard to the different architectures. Another measure of success would be our ability to attack the efficacy of the motion forecasting predictions using different adversarial attack methods.

#### IF YOU ARE DOING SOMETHING NEW, EXPLAIN HOW YOU WILL ASSESS YOUR MODEL’S PERFORMANCE.

There are various ways to measure motion forecasting accuracy: Miss rate: whether the predicted end point is a significant distance from the actual end point L2 distance between the forecasted trajectory and the actual trajectory We will report a selected range from these different accuracy metrics.

For adversarial attacks, we may qualitatively compare inputs based on the attack and real-world inputs, then quantitatively assess the difference in prediction accuracy between the original and adversarially perturbed input.

#### WHAT ARE YOUR BASE, TARGET, AND STRETCH GOALS?

The base goal is to implement several baseline motion prediction models which achieve reasonable accuracy (and pass an “eye test”) on the argoverse 2 motion prediction dataset. The target goal is to select the most promising one (could be one of ours, or a separate, more serious outside implementation), and apply adversarial attacks to it and demonstrate severe accuracy decline on imperceptibly perturbed trajectories. Finally, the stretch goal will be to modify a motion prediction model to be more robust to adversarial perturbations.

### Ethics

> Choose 2 of the following bullet points to discuss; not all questions will be relevant to all projects so try to pick questions where there’s interesting engagement with your project. (Remember that there’s not necessarily an ethical/unethical binary; rather, we want to encourage you to think critically about your problem setup.)

#### WHAT BROADER SOCIETAL ISSUES ARE RELEVANT TO YOUR CHOSEN PROBLEM SPACE?

Motion forecasting has direct relevance to the realm of autonomous vehicle development. There are many sides to the debate around safety, accountability, and agency in this space. There are also things that can be learned in the areas of human behavioral analysis where exploring this type of problem can uncover how and why humans handle certain driving situations.

#### WHY IS DEEP LEARNING A GOOD APPROACH TO THIS PROBLEM?

Motion forecasting has strict real-time requirements, but is an endlessly flexible and “soft” problem, in the sense that very few hard rules can be written that won’t be violated from some complicated dynamic. There is also a proliferation of data collected from self-driving car companies. Due to the lack of easy answers, the magnitude of data that has already been collected, and the fact it can clearly be posed as a time series regression problem, this is a good problem for deep learning. Additionally, unlike some time series problems (where XGBoost may be the best algorithm of choice), there is also a significant geometric structure to the data which may suit certain neural network architectures.

#### WHAT IS YOUR DATASET? ARE THERE ANY CONCERNS ABOUT HOW IT WAS COLLECTED, OR LABELED? IS IT REPRESENTATIVE? WHAT KIND OF UNDERLYING HISTORICAL OR SOCIETAL BIASES MIGHT IT CONTAIN?

The datasets Argoverse 1 and Argoverse 2 are collected by a single company. It may work the best only on the company’s self-driving systems.

Argoverse 2 is collected from Austin, Detroit, Miami, Palo Alto, Pittsburgh, and Washington D.C. Rural areas may be less represented. Thus, any results found are going to be most beneficial and constrained to urban areas, especially companies operating in those areas.

#### THE STAKEHOLDERS

> Who are the major “stakeholders” in this problem, and what are the consequences of mistakes made by your algorithm? How are you planning to quantify or measure error or success? What implications does your quantiﬁcation have? Add your own: if there is an issue about your algorithm you would like to discuss or explain further, feel free to do so.

The stakeholders of the problem include employees at Argo, the broader research community working on motion prediction, potential customers of self-driving car companies, and engineers who work on self-driving car components which interact with and may rely on safe motion prediction. By attempting adversarial attacks on the model, we can demonstrate a lack of robustness, which could have safety implications.

### Division of labor

> Brieﬂy outline who will be responsible for which part(s) of the project.

The group will collaboratively choose a different set of hyperparameters for GRU and LSTM. Once we find a method to attack the model, the group will use a similar strategy to tune the hyperparameters.