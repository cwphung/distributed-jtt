# Distributed JTT

## JTT Background
JTT is a two-stage training algorithm used to mitigate spurious correlations in a dataset. It involves performing several epochs of empirical risk minimization (ERM) training and clustering datapoints into majority (correctly classified) and minority (incorrectly classified) groups. This step is called group inference. The minority group is then upsampled with respect to the ratio of majority group samples to minority group samples, and the augmented dataset is used to train a new ERM model. This second step is called robust training.

An example application of this algorithm involves the Waterbirds dataset, which seeks to differentiate land birds from water birds. Since most water birds are pictured around water and land birds are pictured around land, a neural network may instead learn to differentiate water images from land images since that is an easier task than differentiating different types of birds.

![An-illustration-of-the-Waterbirds-dataset-A-spurious-correlation-exists-between-the-bird](https://github.com/user-attachments/assets/e4b9d89f-7aa7-4e69-95b6-f7f748c75a2c)

JTT addresses this spurious correlation by upsampling the misclassified examples after initial training (water birds at land, land birds at water) then retraining on the augmented dataset. Since the dataset now has an equal number of waterbirds at land and waterbirds at water, it is forced to learn to differentiate the land birds from water birds.

## Distributed Approach
We analyze the effects of distributing the group inference stage of JTT across multiple machines.

Group Inference
1. Shuffle data and evenly distribute between N machines
2. Train each machine on its received subset of data for 2 epochs
3. Divide each received subset into majority and minority group based on initial output classification
4. Combine minority groups of each data subset into a total minority group
5. Combine majority groups of each data subset into a total majority group

Robust Training
1. Define nominal upsample as the number of items in total majority group divided by number of items in total minority group
2. Upsample minority group by floor of 0.8x nominal upsample
3. Train on augmented dataset for 3 epochs

## Results
Results are derived from training on the SPUCOMNIST Dataset, which contains written numbers with a colored box at the upper right corner of the image representing the spurious feature.

![af3c6733-4925-4bcf-a232-96de118c5ea1](https://github.com/user-attachments/assets/9c0e4e82-be6f-48f4-b62d-5a991a7dbd43)

The worst-group accuracy decreases as the number of machines used for distributed JTT increases. This outcome is expected because when the dataset is distributed, the group inference performed on each machine's subset becomes less accurate (each machine does not have full context of the entire dataset). The aggregate minority group after group inference is effectively a noisy approximation of the actual minority group. We notice that distributed JTT does not have significant performance dropoff when distributed across 10 or fewer machines.

## Future Work
Our JTT training parameter choices (e.g. num layers, num training epochs) can be optimized to improve base (non-distributed) worst-group accuracy. Our base accuracy is around 30% while other studies have achieved up to 74% accuracy. In addition, the trade-off between JTT accuracy and computational speed-up may be different for more complex datasets (e.g. Waterbirds). It is possible that especially complex datasets may not tolerate dataset splitting because performing effective group inference requires knowing the full dataset context.
