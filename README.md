# choosingClusteringAlgorithms
Example implementation of an algorithm to choose the 'optimal' clustering algorithm out of a slim pool of clustering algorithms based on quick-to-calculate metrics using Northwind data. The goal is to perform efficient calculations on the data to determine the clustering algorithm that best fits the distribution and then using that algorithm to perform the actual clustering task. Currently, this is done by giving a rating to each candidate based on 5 scores:
1. __Deviation Score__: Evaluates the standard deviations σx and σy in relation to their respective value range of the parameters x and y.
2. __Distribution Score__: Obtains information about the density of the distribution by determining the percentage of elements for each of the axes x and y that lay within bσ range of the median for x and y respectively, where b∈Q. Also  obtain information about equipartition of the distribution by counting the values in +σ-range of the medians of x and y and comparing the results with those of the −σ-range of the medians of x and y respectively. The result is composed of 4 values that make up the score: s<sub>dist</sub> = (s<sub>xr</sub>, s<sub>xl</sub>, s<sub>yr</sub>, s<sub>yl</sub>)
3. __Equilibrium Score__: Get the (−1,1) normalized position of the median for both axesx and y. The combination of both of these values indicates towards which quadrant the distribution is shifted where the point of origin is to be seen as the point that marks the middle m of the value ranges for both x and y respectively P(x<sub>m</sub>/y<sub>m</sub>) = (0/0). The resulting point P(x<sub>med</sub>/y<sub>med</sub>) is correlated to the score: seq=P(x<sub>med</sub>/y<sub>med</sub>) where x<sub>med</sub> and y<sub>med</sub> are  the normalized medians of x and y	respectively.
4. __Repetition Score__: Count repetitive values in the data set, unrelated to where they occur, and determine the ratio of repetitions s<sub>rep</sub> in the dataset compared to the number of total values scaled to (0,1) for each parameter. We  define s<sub>rep</sub> as the combination of the normalized scores for both axes: s<sub>rep</sub>= (s<sub>xrep</sub>, s<sub>yrep</sub>)
5. __Sparse spot detection__: Take the standard deviations σ<sub>x</sub> and σ<sub>y</sub> and find sparse areas within the data by checking the vicinity for each value for x and y with their respective standard deviation. If too few/ too many spots are found increase/ decrease the vicinity by introducing the factors n ,m to get nσ<sub>x</sub> and mσ<sub>y</sub>. Repeat until an acceptable amount of holes have been found or too many iterations in one direction have been made. For instance, decreasing n or m 3 times in a row could be a terminating condition. In that case, stop and proceed by assuming no sparse areas.
# Languages, Frameworks
The application was written in Python using the plotly-dash framework to create a web interface to manipulate the inputs and see the results. 
The original use case is bank customer data with millions of rows of data but for the sake of demonstration in this open source repository, Northwind data is used.
# How to use
In order to use the application, a data source needs to be provided. The application is exemplarily set to use data from a local mssql northwind data base that can be created by running the northwind.sql script that resides in the repository, or by following this guide: https://parallelcodes.com/install-and-attach-northwind-database-in-sql-management-studio/ 

After Launching (I suggest Debug mode, since it's easier to shutdown), the web interface will be reachable at http://127.0.0.1 as is displayed in the console. In its current state, the application will take a few seconds to few minutes to display results based on the chosen parameterization. At the top of the Webpage, the chosen algorithm can be seen.
# How to contribute
With the somewhat modular implementation of scores, additional features can be added, scores can be modified or removed etc. without much changing of the rest of the algorithm. Some of the scores of the current implementation lack use or need to be reworked. 
# How to cite
If you use this algorithm or parts of it, or if you reference it in a paper, please cite it in your publication.
```
@software{ht2021chalg,
  author       = {Hardy Taulien},
  title        = {An algorithm to optimize choosing of a clustering algorithm},
  month        = march,
  year         = 2021,
  url          = {https://github.com/hardy-taulien/choosingClusteringAlgorithms}
}
```
