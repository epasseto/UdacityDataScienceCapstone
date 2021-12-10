# Udacity Data Science Course Capstone

## Dog Breeding Identification Project

#### Objective:

The main idea to this project is to have contact with **Perceptrons** technology.

The theoretical concepts involved in **Perceptrons** are not so new. I fact, **Frank Rosenblatt** was evolving that was known as the **first** Perceptron in the history, in 1958, for military applications (source, [Wikipedia](https://en.wikipedia.org/wiki/Perceptron)).

With the evolving of computers, specially **GPUs** (Graphic Processor Units), the search for machineas that could emulate the working of a **neuron** was intensified. The proposal is, with **minimal** human intervention, to train a machine that can make tasks involving **Classification** or **Fitting Curves** at least so well as humans can do. Perceptrons evolved in a way that they proved to be a great alternative for some kinds of problems, as **Image Classification**.

The task proposed by **Udacity** as Capstone for **Data Science Nanodegree** was to make some experimentation involving **Perceptrons**. Perceptrons are also known as a **Deep Learning Technique**, what means that it involve more than one layer of learning.

Udacity provided for this project, a **Jupyter Notebook** and **Extra Classes** as part of the learning process. The programming language involved is **Python**, and also some specific libraries, as **Pandas** (for dataframes opperations), **Keras** (for Deep Learning machines), and **Numpy** (for mathematica/geometrical operations), just to cite some of them.

---

## Metodology

Udacity metodology is strongly inspired by the formula **DCR** (**Drive, Correct, Repeat**). This small acronym refers for the learning process involved in training a new car driver:

> - first, the instructor gives you some **orientation** about safety rules and the tasks you will need to do, and then letting you to try by yourself (**Drive**);
> - second, you will be **corrected** from your mistakes and how to avoid them. If is necessary, new explanations will be given for a better drive experience (**Correct**);
> - third, you will be oriented to try again, and again, until you achieve the necessary **expertise** for driving a car by yourself (**Repeat**).

As an **analogy** to learning how to drive a car, some explanation is given in an **incomplete** Jupyter Notebook. Explanatory notes will be found in all your learning process. And if you stuck in some point, you can as for **Mentor** help. I made it a lot of times. And having a **Mentor** (not quite a **Mentat**, like you can see in this Sci-Fi movie **Dune** - 2021) is really a good thing, when you are lost in a forest of **Coding** trees!

---

## Parts of the project

#### Step 0: Import Datasets

Also I needed to import my libraries and was asked to create/evolve some predefine **functions** for making everything running well. It was not so hard!

#### Step 1: Detect Humans

The idea was to use as base a OpenCV (a public domain face detector) and a bunch of pre-loaded data. And then was invited to create my own **Human Face Dectector**, giving a picture do say if a human was there or not.

#### Step 2: Detect Dogs

As this is a project for **Dog Breeding Detection**, the next step was to make the same thing, involving now images and evolving a pre-trained **Dog Detector**. The involved technology for both detectors was **CNN** (**Convolutional Neural Networks**). Pre-processing phases involved turning a Image into a Tensor, evolving functions to turn this step easier. Predictions were made using a pre-trained CNN named **ResNet-50**. Then I was invited to write my own **Dog Detector**.

#### Step 3: Create a CNN from Zero

In this phase, I was asked to create my own **CNN** to classify **Dog Breeds**. In this time, from Scratch. That means that I will not be using third-part ready-made technology. I needed to crate my own **Architecture** and was asked for **at least 1%** of Accuracy on the **Test** dataframe. It sounds as a **easy** task, but it is not so! Some dog breads are some similar from one to another that it is really **hard**, even for me, to tell exactly what the breed is, giving a picture.

Some adjustments were suggested (my first **Accuracy** was only 0,93%), like to add **Dropout** Layers, for preventing **Overfitting** during the Train phase. To determine the best number of **Epochs** (machine states, that creates a model to be evaluated), mas for me, a bit **challenging**. My best Accuracy was near to **12%**. Not so bad, for a written from zero model!

#### Step 4: Use a CNN to Classify Dog Breeds

The idea in this step was to use pre-trained **Bottlenecks** and then just adding new layers form zero, to be trained. It sounds as a **strange** idea, as we were using first Perceptron Layers trained for other uses (as detecting **Human**) faces for a dog-detection task, but it makes some sense:

>- general patterns are **universal**, so the first layers of a Perceptron could detect, by long training, to capture skin texture, background differentiation, horizontal and vertical lines, etc..

>- pictures containing **faces**, being from a human, or a dog, have some similarities. As the general shape, the contrast between the face-area and the background, etc..

>- a so-well trained **Bottleneck** was evolved so much, trained with hundreds of thousands of images, even **millions** of images, that the captured patterns are really reliable!

I was asked to complete the Model Architecture, adding some Layers by myself. This demmanded a bit of **study** of what these Layers could do for me, the **dimmensions** involved, and a lot of **parameters** to set. The model was then compiled, trained and checked for Accuracy on the **Test** Dataframe. Some **Dog Breeds** were predicted with the new Model, now with a Test Accuracy near to **40%**.

#### Step 5: Create a CNN to Classify Dog Breeds

This time the tactic evolved to using **Transfer Learning**. The expected Accuracy was at least **60%** at the Test Dataframe. We were invited to experiment with some Bottlenecks available in Keras (some of them earned very **notable** prizes!). Whe could use things like VGG-19, ResNet-50, etc..

I obtained some of these Bottlenecks (not all of them, as we had limited **GPU Time** and Storage Capacity on Udacity servers). Then, I created a Function for making adjustments and iterating with my chosen Bottlenecks (**ResNet-50** and **InceptionV3**). After a quite hard job, I could at the same time, evolve a better **Tester/Feeder** Function, that I named **The Charger**, and obtain the best parameters that I could find for attaining near to **75%** of Accuracy on Test Dataframe. Not bad at all!

For **fun** I took one picture of myself and tested on my already trained Classifier. And if I was a dog, my breed should be **Silky Terrier**! Oh, it was a really **good momen** for a joke!

#### Step 6: Write your Algorithm

The idea now is to create a Application by myself. So, taken a pre-trained Classifier and some pictures in **public domain** and the ones already preexistent in a Udacity folder to say:

>- if there is a **dog** in the picture;
>- if there is a **human** in the picture;
>- if a **human** was found, what it will be his most likely **dog breed** it it was a **dog**;
>- give error warnings, if something went wrong.

#### Step 7: Test your Algorithm

1. every **dog** I tested, from Udacity folder, was recognized as a dog and **only** as a dog (not a human). Nice;

2. some of the "humans" that I tested form free domain images, were tested as **humans** (Einstein picture, Mr Spock from Star Treck, Roland Macdonalds and even Fofão - a man dressed as a dog, very famous in Brazillian culture). They also passed in the test to **not** being a dog, and received some (only for **humor**!) dog breeds tags if they were dogs, what breed they could be;

3. two of the "humans" didn´t pass in the **human** test. One was a picture from **Werewolf**, a 1995 movie and for (**obvious**!) reasons, the monster was **not** interpreted as a human! The other one was my own picture and the most likely reason is that it is a so **distorted** pic, taken with my old webcam, and so **badly** reshaped for 224x224 image that the Perceptron could not recognize it as a human. OK, I can pass with this idea!  

---

### Involved files:

- the `dogg_app.ipynb` is the most important file involvend in this project.

- if you really want to run it, you will need a computer with `Jupyter Notebook` pre-installed and an active **GPU** to run it;

- the other files that are not provided in this **GitHub**, you will need to donwload them from the propper sources, in a way to run this project.

---

### Versions:

- 0.1-0.2 → october, november 2021

- 1.0 → november, 27, 2021

... (a bunch of mistakes corrected)

- 1.4 → december, 10, 2021

