# capstone_design

11차례의 accuracy, loss graph 실험중 image augmentation을 통해 각각 300장의 이미지를 2000개씩으로 늘려 validation set 95% accuracy를 달성하였다.
그러나 실제 테스트를 해본결과 validation set에 overfitting 되어있어서 제대로 동작하지 않았다.

# over fitting 된 accuracy, loss
![8_acc](https://user-images.githubusercontent.com/70372577/140053082-7d69b308-e0c2-4ea6-9423-b98408dcc25e.png)
![8_loss](https://user-images.githubusercontent.com/70372577/140053092-cfe74262-f989-4daa-b1e2-93e8747652f2.png)

# augmentation을 하지않고 학습시킨결과
![9_acc(without augmentation)](https://user-images.githubusercontent.com/70372577/140053210-dd00a6ec-1048-49b4-a3d7-45c9904b3b78.png)
![9_loss(without augmentation)](https://user-images.githubusercontent.com/70372577/140053212-0f1420ef-79d9-4da5-9adf-8b905aede824.png)

이 프로젝트는 실제데이터 부족으로 실패했던 사례이며, 다음부터는 실제데이터에 집중하되 
validation set에 over fitting 되지는 않았는지 살펴보아야 할 필요가 있다.
