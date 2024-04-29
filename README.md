你好，我有一些关于上学期AI这节课的成绩的疑惑。关于您给我的评语中写到的问题，我想给你解释一下。可能是我提交的作业中标注的不太清楚，所以我整合了一下我项目中所做的工作，希望你帮我看一下是不是可以能不能提高该门课的成绩。

The following is an introduction to the modified parts of the code in my project

Hyperparameter section:
First, set the number of categories to 5, because there are five artistic styles in total. Secondly, in order to verify and test the performance of the model, the data set is divided into a training set, a verification set, and a test set.num_classes = 5
val_size = 0.2
test_size = 0.2
batch_size = 150
learn_rate = 0.001(Different values were tested)

Preprocessing and data enhancement part:
Since the data enhancement in the original case was already relatively comprehensive, not many other operations were added.
Currently the following two have been added
# Flip the image vertically with 50% probability to increase the model’s robustness to up and down flips
transforms.RandomVerticalFlip(p=0.5),
# Perform a random rotation between -90 and 90 degrees
transforms.RandomRotation(degrees=(-90,90)),

Dataset loading part:
In order to facilitate testing the model effect, the code for dividing the test set is added.
The following is the corresponding code
# First split for train and temporary split. Test size here is combined validation and test size
train_indices, temp_indices = train_test_split(indices, test_size=val_size + test_size, random_state=42)

# Second split for validation and test from the temporary split
val_indices, test_indices = train_test_split(temp_indices, test_size=test_size/(val_size + test_size), random_state=42)

Visualize parts of the dataset:
Added code for visual test set
# Plot some test images
real_batch = next(iter(test_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Test Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

Model part:
I reduced the number of filters in the first two convolutional layers, which reduces the model size and computational requirements. At the same time, by adding padding in the convolution operation, it can be ensured that the size of the feature map is maintained after the convolution operation. A Dropout layer is added to prevent model overfitting. The number of neurons in the first fully connected layer is returned to 512 to reduce model complexity and increase training speed. If the above effect is not very good, I also tested the more complex model written below, but found that the speed would be slower.

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 8 * 8)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        return x   

Test part:
In addition, after training, the test code is also added. The main indicators include Confusion Matrix, Classification Report, and Accuracy.


About data selection： 
The source of the data set is crawled from the Pinterest website, and the data type is art pictures of different categories. I grabbed a total of 5 pictures of artistic styles and put them into classes 1-5 respectively. At the beginning, 
I found in the test that the selection styles of the data sets were relatively similar, but the current model could not classify types that were too similar, and the results were not ideal, so I increased the amount of data and chose a more distinctive artistic style for classification. 
The data Three of the groups were concentrated at 3000-4000 pictures, and the other two groups were 716 and 1476. I entered more data into the groups whose styles were more difficult to distinguish. In addition to data preprocessing, I also performed manual filtering.
