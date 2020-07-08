# How to predict movies with K-Nearest-Neighbors(KNN)

> Subtitle or Short Description Goes Here
> movies, K-Nearest-Neighbors,KNN, predict movies
<!-- [![ThisIsJorgeLima](https://miro.medium.com/max/1024/1*dU4oVsAravaoBRaskvUA9A.jpeg)](https://www.thisisjorgelima.com/2020-05-29-How-to-predict-movies-withk-nearest-neighbors/) -->


How many of you spend countless hours trying to find the right movie to watch? I second that! The premise of this build was to predict recommendations from user ratings that have rated 75 or great films. By inputting a film title into our application we used the k-nearest neighbors aka KNN algorithm that predicts an output of ten recommendations. I’ll do my best to break down the process and the thinking in layman terms as best I can.

**Setup:**


```
Jupyter notebook
text editor
or
Google Colab
```


**Installation:**

```
pip install scikit-learn
pip install fuzzywuzzy
pip install python-Levenshtein
pip install sparse==0.1.1
```

**Import Statements:**
```
#classics
import pandas as pd
import numpy as np
import os

# algorithm
from sklearn.neighbors import NearestNeighbors
# sparse matrix

from scipy.sparse import csr_matrix

# string matching
from fuzzywuzzy import fuzz
from Levenshtein import *
from warnings import warn
```

**Environment setup:**

1. Open your command-line (Terminal for MacOS, Anaconda Prompt for Windows)
2. Navigate and download [requirements.txt](https://www.dropbox.com/s/i1vjd48ww1lvdye/requirements.txt?dl=0)
3. Run ```conda create -n FILMLY python == 3.7```
 Once the command completes, your conda environment should be ready to go.
4. Now, we are going to add the proper python packages for this application. You will need to activate your conda environment: ```
source activate FIMLY```
on Terminal or ```
conda activate FILMLY```
 on Anaconda Prompt. Once your environment is activated, run ```
pip install -r requirements.txt```
, which will install the required packages into your environment.

5. We are also going to add an ipython Kernel reference to your conda environment, so we can use it in JupyterLab or text editor of choice.

6. Next, we will run ```
python -m ipykernal install — -user — -name FILMLY- -display-name “FILMLY(Python3)"```
 => This will add a JSON object to an ipython file. So when we are using JupterLab if you go this route. It will know that it can use this isolated instance of Python.

7. Deactivate your conda environment and launch JupyterLab. It would help if you remembered to see “FILMLY (Python3)” in the list of available kernels on the launch screen.(Jon-Cody Sokoll)

After installing the above assets to build our project we can download our dataset from [MovieLens](https://grouplens.org/datasets/movielens/latest/). This data changes over time, as listed on site. “it is not appropriate for report research results.” This zip file contains several files. We are only utilizing movies.csv and ratings.csv. There is a tag DataFrame that will be interesting to implement into another build. For now, I will stick with those two files.

Now we can load and validate our data.

```
# configure file path
data_path = os.path.join(os.environ['DATA_PATH'], 'MovieLens')
movies_path = 'movies.csv'
ratings_path = 'ratings.csv'#read data
movies = pd.read_csv(
    os.path.join(data_path, movies_filename),
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})

ratings = pd.read_csv(
    os.path.join(data_path, ratings_filename),
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
```
Next will view our movies data:
[](https://miro.medium.com/max/920/1*zft1nmNQ1HPA1nbk90LzIw.png)

Now our rating data:
[](https://miro.medium.com/max/616/1*N9_Kmx90LCJTXzYbZTJ5IQ.png)

```
print("Top 10 Rating by user")
print(df["rating"].value_counts().head(10))

plt.figure(figsize=(8,6))

#Total rating distribuition
g = sns.distplot(df["rating"], bins=20)
g.set_title("Rating distribuition", size = 20)
g.set_xlabel('Rating', fontsize=15)
```


Now that both of our datasets we have to pivot the DataFrame and also make sure fo fill NA/NaN using a specified method. We use this to fill any holes in our data a simply default the value to zero. Don’t worry any values, not in our DataFrame will not be filled. This way That way it doesn’t interfere:
```
# Here we are making a pivot to set our dataset and features
features = ratings.pivot(index='movieId', columns='userId',  values='rating').fillna(0)
# now will convert our DataFrame of movie features to scipy sparse matrix:
matrix_movie_features = csr_matrix(features.values)
# number of ratings each movie recived:
movie_count = pd.DataFrame(ratings.groupby('movieId').size(), columns=['count'])
```
[Our DataFrame Matrix.](https://miro.medium.com/max/1400/1*AOB9Rk5FdWG5hdOhy4eS4w.png)

Now we can discuss how we are pivoting into [K-Nearest Neighbors(K-NN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). Why do you ask why we did an array? For our film recommended we needed our data to be in anm ```x n``` array. Our``` m ```represents the number of moves and ``` n``` represents our number of users. We had to reshape our DataFrame by ```pivot``` which allows setting the movie as rows and format our users as columns. Since our rating distribution is 0–5 as the chart below. We will be calculating the distances between our vectors. By doing this will need to perform linear algebra. (Kevin Liao)

[](https://miro.medium.com/max/966/1*OH9B6CFcuJCAPrrIKN6LIQ.png)

We will be using cosine similarity for our nearest neighbor search. When a KNN helps us under the hood, it is relying on us to set up our pivot and features to help predict movies by calculating our distances between similar user movie recommendations reason why are dataset is sparse.

[](https://miro.medium.com/max/1280/1*0MSjneF0qtYaxLUgsw2Ybw.gif)
*Here is an example of a KNN with 5 neighbors. We will set our parameters to 10 so we can predict ten movies. To learn more I recommend watching [StatQuest: K-nearest neighbors, Clearly Explained](https://www.youtube.com/watch?v=HVXime0nQeI&feature=youtu.be)*

**Parameters:**

**n_neighbors: int, default = 5**

*   Will set ours to 10
**algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default = ‘auto’**

*   optional metric: string or callable, default ‘minkowski’
*   ‘ball_tree’
*   ‘kd_tree’
*   ‘brute’ will use a brute-force search.
*   ‘auto’

Read more on [```sklearn.neighbors```](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors).KNeighborsClassifier
Lets review. Now we have our sparse matrix complete our DataFrame is transformed and ready to predict movies. We will now need to implement our KNN model with the following parameters:
```
# applying and defining our model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
#fit
model_knn.fit(movie_user_matrix_sparse)
```

Since we will not be utilizing NLP natural language processing. We, will, kind of, by using a magical matcher called FuzzyWuzzy. This will aid us in getting a proper ratio of our movie names by string matching. Under the hood, it uses [Levenshtein](https://en.wikipedia.org/wiki/Levenshtein_distance) Distance which will calculate between sequences.

```
def recommendation(model_knn, data, mapper, favorite_movie, n_recommendations):

 # fit
 model_knn.fit(data)
 # get our movie index
 print(‘Film input’, favorite_movie)
 index = fuzzy_matcher(mapper, favorite_movie, verbose=True)
print(‘Popular recommendations:’)
 print(‘…..\n’)
 distances, indices = model_knn.kneighbors(data[index], n_neighbors=n_recommendations+1)
raw_recommends = sorted(
 list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
 # reverse mapping and unflattening
 reverse_mapper = {v: k for k, v in mapper.items()}
 # print recommendations
 print(‘Here are more movies similar {}:’.format(favorite_movie))
 for i, (index, dist) in enumerate(raw_recommends):
 print(‘{0}: {1}, with distance of {2}’.format(i+1, reverse_mapper[index], dist))
 ```
Now we can make movie predictions.

1. Open your command-line (Terminal for MacOS, Anaconda Prompt for Windows)
2. now run this command: ```python or python3 [FILENAME].py```

let's see our results:
[](https://miro.medium.com/max/1400/1*WJOAz-seCuZ7fWHJTT27cw.png)

And Wallah! We have our ten predictions! Wait for one minute. If you’d like to have fun change colors within your command line or even add a title card to your application like:

[](https://miro.medium.com/max/1400/1*0MVgasqdEWtpdvMyhF8wpw.png)

 ```

def displayIntro():
    print(" Find a film to watch.")
#
# Main
#
# Clears screen for opening logo :
clear_terminal()
# change colors:
# text = colored('Hello, World!', 'red', attrs=['reverse', 'blink'])
# print(text)
print(Fore.CYAN)
print("""
███████╗██╗██╗░░░░░███╗░░░███╗██╗░░░░░██╗░░░██╗
██╔════╝██║██║░░░░░████╗░████║██║░░░░░╚██╗░██╔╝
█████╗░░██║██║░░░░░██╔████╔██║██║░░░░░░╚████╔╝░
██╔══╝░░██║██║░░░░░██║╚██╔╝██║██║░░░░░░░╚██╔╝░░
██║░░░░░██║███████╗██║░╚═╝░██║███████╗░░░██║░░░
╚═╝░░░░░╚═╝╚══════╝╚═╝░░░░░╚═╝╚══════╝░░░╚═╝░░░
                                © MMXX JRGL.IM
    """
      )
displayIntro()
type_and_will_predict_favorite_films = input(' Search now.. \n')
 ```

First, you’ll have to make a new filename. I simply named it utils.py

Then add this code in your new file:

 ```
"""
Import Statements:
"""
from os import name, system
def clear_terminal():
    if name == "nt":
        # Windows
        system("cls")
    else:
        # Linux or Mac
        system("clear")

 ```

Second, go back to your application.py or [filename].py you named your project and add the following in your **Import Statements:**

 ```

"""
Import Statements:
"""
# change font colors within command line
from colorama import Fore, Back, Style

# clears our terminal for title
from utils import clear_terminal #. where utils add [filename].
 ```

Add the code above and you’ll have an interacting interface to search for your films. If you’d like to learn more about how to customize and [print colors](https://www.geeksforgeeks.org/print-colors-python-terminal/) in Python terminal here’s a guide.

In conclusion, coming from a film background and true film buff this was a fun build. It’s not perfect and does occasionally make errors. I may need to modify the FuzzyWuzzy and do a straight-up Natual Language Processing, to help implement get us better analysis. Which will include a corpus of text using text visualizations of token frequency in regards to our movie titles.
That will be for another blog another day. 

### want to get in touch?

*  [Linkedin](https://www.linkedin.com/in/jorgelima/)
*  [Twitter](https://www.twitter.com/thisisjorgelima/)
*  [Podcast](https://mailchi.mp/db9640dec7a5/a-month-of-saturdays)
*  [Instagram](https://www.instagram.com/thisisjorgelima/)
*  [Facebook](https://www.facebook.com/thisisjorgelima/)
*  [Website](https://www.thisisjorgelima.com/)
*  [Github](https://www.github.com/thisisjorgelima/)

### You can applaud my story on Medium here:
[Medium](https://medium.com/@ThisIsJorgeLima/how-to-predict-movies-with-k-nearest-neighbors-knn-714aaaf12791)

[**GitHub Repo.**](https://github.com/ThisIsJorgeLima/CS-Data-Science-Build-Week-1/tree/jorge-lima)

**Resources:**

[MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/latest/),
[k -nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm),
[A Gentle Introduction to Sparse Matrices for Machine Learning](https://machinelearningmastery.com/sparse-matrices-for-machine-learning/),
[Here’s How to Build a Pivot Table using Pandas in Python](https://www.analyticsvidhya.com/blog/2020/03/pivot-table-pandas-python/),
[Building a movie recommendation system K-Nearest Neighbors Machine Learning](https://www.youtube.com/watch?v=4Ws0oPH350U),
[Machine Learning Basics with the K-Nearest Neighbors Algorithm](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761),
[Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering),
[Prototyping a Recommender System Step by Step Part 1: KNN Item-Based Collaborative Filtering:](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761),
[Fuzzy String Matching in Python](https://www.datacamp.com/community/tutorials/fuzzy-string-python),
[Reverse Mapping and Unflattening](https://docs.automapper.org/en/stable/Reverse-Mapping-and-Unflattening.html),
[Comprehensive Guide to building a Recommendation Engine from scratch (in Python)](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/?#)


**Badges will go here**

- build status
- issues (waffle.io maybe)
- devDependencies
- npm package
- coverage
- slack
- downloads
- gitter chat
- license
- etc.

***INSERT ANOTHER GRAPHIC HERE***

[![INSERT YOUR GRAPHIC HERE](http://i.imgur.com/dt8AUb6.png)]()

- Most people will glance at your `README`, *maybe* star it, and leave
- Ergo, people should understand instantly what your project is about based on your repo

> Tips

- HAVE WHITE SPACE
- MAKE IT PRETTY
- GIFS ARE REALLY COOL

> GIF Tools

- Use <a href="http://recordit.co/" target="_blank">**Recordit**</a> to create quicks screencasts of your desktop and export them as `GIF`s.
- For terminal sessions, there's <a href="https://github.com/chjj/ttystudio" target="_blank">**ttystudio**</a> which also supports exporting `GIF`s.

**Recordit**

![Recordit GIF](http://g.recordit.co/iLN6A0vSD8.gif)

**ttystudio**

![ttystudio GIF](https://raw.githubusercontent.com/chjj/ttystudio/master/img/example.gif)

---

## Table of Contents (Optional)

> If your `README` has a lot of info, section headers might be nice.

- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [Team](#team)
- [FAQ](#faq)
- [Support](#support)
- [License](#license)


---

## Example (Optional)

```javascript
// code away!

let generateProject = project => {
  let code = [];
  for (let js = 0; js < project.length; js++) {
    code.push(js);
  }
};
```

---

## Installation

- All the `code` required to get started
- Images of what it should look like

### Clone

- Clone this repo to your local machine using `https://github.com/fvcproductions/SOMEREPO`

### Setup

- If you want more syntax highlighting, format your code like this:

> update and install this package first

```shell
$ brew update
$ brew install fvcproductions
```

> now install npm and bower packages

```shell
$ npm install
$ bower install
```

- For all the possible languages that support syntax highlithing on GitHub (which is basically all of them), refer <a href="https://github.com/github/linguist/blob/master/lib/linguist/languages.yml" target="_blank">here</a>.

---

## Features
## Usage (Optional)
## Documentation (Optional)
## Tests (Optional)

- Going into more detail on code and technologies used
- I utilized this nifty <a href="https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet" target="_blank">Markdown Cheatsheet</a> for this sample `README`.

---

## Contributing

> To get started...

### Step 1

- **Option 1**
    - 🍴 Fork this repo!

- **Option 2**
    - 👯 Clone this repo to your local machine using `https://github.com/joanaz/HireDot2.git`

### Step 2

- **HACK AWAY!** 🔨🔨🔨

### Step 3

- 🔃 Create a new pull request using <a href="https://github.com/joanaz/HireDot2/compare/" target="_blank">`https://github.com/joanaz/HireDot2/compare/`</a>.

---

## Team

> Or Contributors/People

| <a href="http://fvcproductions.com" target="_blank">**FVCproductions**</a> | <a href="http://fvcproductions.com" target="_blank">**FVCproductions**</a> | <a href="http://fvcproductions.com" target="_blank">**FVCproductions**</a> |
| :---: |:---:| :---:|
| [![FVCproductions](https://avatars1.githubusercontent.com/u/4284691?v=3&s=200)](http://fvcproductions.com)    | [![FVCproductions](https://avatars1.githubusercontent.com/u/4284691?v=3&s=200)](http://fvcproductions.com) | [![FVCproductions](https://avatars1.githubusercontent.com/u/4284691?v=3&s=200)](http://fvcproductions.com)  |
| <a href="http://github.com/fvcproductions" target="_blank">`github.com/fvcproductions`</a> | <a href="http://github.com/fvcproductions" target="_blank">`github.com/fvcproductions`</a> | <a href="http://github.com/fvcproductions" target="_blank">`github.com/fvcproductions`</a> |

- You can just grab their GitHub profile image URL
- You should probably resize their picture using `?s=200` at the end of the image URL.

---

## FAQ

- **How do I do *specifically* so and so?**
    - No problem! Just do this.

---

## Support

Reach out to me at one of the following places!

- Website at <a href="http://fvcproductions.com" target="_blank">`fvcproductions.com`</a>
- Twitter at <a href="http://twitter.com/fvcproductions" target="_blank">`@fvcproductions`</a>
- Insert more social links here.

---

## Donations (Optional)

- You could include a <a href="https://cdn.rawgit.com/gratipay/gratipay-badge/2.3.0/dist/gratipay.png" target="_blank">Gratipay</a> link as well.

[![Support via Gratipay](https://cdn.rawgit.com/gratipay/gratipay-badge/2.3.0/dist/gratipay.png)](https://gratipay.com/fvcproductions/)


---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2015 © <a href="http://fvcproductions.com" target="_blank">FVCproductions</a>.
 
