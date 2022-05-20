import torchvision
import torch.nn as nn
import torch.nn.functional


class ClassifierModel(nn.Module):
    def __init__(self, net_type: str, num_classes: int):
        super(ClassifierModel, self).__init__()
        self.oriNet = None
        self.get_classification_model(net_type, num_classes)

    def get_classification_model(self, net_type, num_classes):
        model = None
        if net_type == "alexNet":
            model = torchvision.models.alexnet(pretrained=True)
            model.classifier[6] = nn.Linear(4096, num_classes)

        elif net_type == "vgg":
            model = torchvision.models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(4096, num_classes)

        elif net_type == "googleNet":
            model = torchvision.models.googlenet(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        elif net_type == "resNet":
            model = torchvision.models.resnet34(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        elif net_type == "mobileNet":
            model = torchvision.models.mobilenet_v3_large(pretrained=True)
            model.classifier[3] = nn.Linear(1280, num_classes)

        self.oriNet = model

    def forward(self, x: torch.Tensor):
        x = self.oriNet(x)

        outputs = nn.functional.softmax(x, dim=1)

        # scores, preds = torch.max(outputs, 1)

        # results = []
        # for idx in range(0, len(scores)):
        #     results.append({"labels": preds[idx], "scores": scores[idx]})

        # return results
        return outputs


if __name__ == '__main__':
    classifier = ClassifierModel('resNet', 2)
    classifier.train()
    out = classifier(torch.rand((4, 3, 120, 120)))
    print(classifier.parameters())
    print(out)
