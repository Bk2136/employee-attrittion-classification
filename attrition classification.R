# Load libraries
library(tidyverse)
library(janitor)
library(MASS)
library(ggplot2)
library(caret)
library(pROC)
library(e1071) 

set.seed(1234)

# Load dataset
employee <- read.csv("employee attrition.csv")

# Basic checks
str(employee)
summary(employee)
colSums(is.na(employee))
sum(duplicated(employee))

# Clean and encode data
employee <- employee %>%
  distinct() %>%
  dplyr::select(-EmployeeCount, -StandardHours, -Over18, -EmployeeNumber) %>%
  mutate(
    Attrition = ifelse(Attrition == "Yes", 1, 0),
    Attrition = as.factor(Attrition),
    across(where(is.character), as.factor)
  )

# Exploratory Data Analysis (EDA)
ggplot(employee, aes(x = factor(Attrition))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Employee Attrition Count", x = "Attrition (0 = No, 1 = Yes)", y = "Count")
ggplot(employee, aes(x = JobRole, fill = factor(Attrition))) +
  geom_bar(position = "fill") +
  labs(title = "Attrition Rate by Job Role", x = "Job Role", y = "Proportion") +
  scale_fill_manual(values = c("grey60", "firebrick"), name = "Attrition") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(employee, aes(x = factor(Attrition), y = MonthlyIncome)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Monthly Income by Attrition", x = "Attrition", y = "Monthly Income")
ggplot(employee, aes(x = OverTime, fill = factor(Attrition))) +
  geom_bar(position = "fill") +
  labs(title = "Attrition Rate by Overtime Status", x = "OverTime", y = "Proportion") +
  scale_fill_manual(values = c("darkgreen", "red"), name = "Attrition")
ggplot(employee, aes(x = Age, fill = factor(Attrition))) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
  labs(title = "Age Distribution by Attrition", x = "Age", y = "Count") +
  scale_fill_manual(values = c("skyblue", "salmon"), name = "Attrition")


employee_encoded <- employee %>%
  dplyr::select(-Attrition) %>%
  model.matrix(~ . -1, data = .) %>%
  as.data.frame()
employee_encoded$Attrition <- employee$Attrition

# Train-test split
set.seed(123)
train_index <- createDataPartition(employee_encoded$Attrition, p = 0.7, list = FALSE)
train <- employee_encoded[train_index, ]
test <- employee_encoded[-train_index, ]
### GLM MODEL ###
glm_model <- glm(Attrition ~ ., data = train, family = binomial)
stepwise_glm <- step(glm_model, direction = "both", trace = FALSE)
# Summary & Odds Ratios
summary(stepwise_glm)
odds_ratios <- exp(coef(stepwise_glm))
print(odds_ratios)
glm_probs <- predict(stepwise_glm, newdata = test, type = "response")
glm_pred <- ifelse(glm_probs > 0.5, 1, 0)
# Confusion Matrix 
confusionMatrix(
  factor(glm_pred, levels = c(0, 1)),
  factor(test$Attrition, levels = c(0, 1))
)
roc_glm <- roc(as.numeric(test$Attrition), glm_probs)
plot(roc_glm, col = "darkgreen", main = "ROC Curve - GLM")
auc_glm <- auc(roc_glm)
cat("GLM AUC:", auc_glm, "\n")


### SVM MODEL ###
train$Attrition <- as.factor(train$Attrition)
test$Attrition <- as.factor(test$Attrition)
svm_model <- svm(Attrition ~ ., data = train, kernel = "linear", probability = TRUE)
svm_pred <- predict(svm_model, newdata = test)
svm_pred_prob <- attr(predict(svm_model, newdata = test, probability = TRUE), "probabilities")[, "1"]
# Confusion Matrix 
confusionMatrix(svm_pred, test$Attrition)
# ROC for SVM
roc_svm <- roc(as.numeric(test$Attrition), svm_pred_prob)
plot(roc_svm, col = "blue", main = "ROC Curve - SVM")
# AUC for SVM
auc_svm <- auc(roc_svm)
cat("SVM AUC:", auc_svm, "\n")


# Load libraries
library(keras)
library(tensorflow)
library(tidyverse)
library(caret)
library(pROC)
# Set seed
set.seed(1234)
employee <- read.csv("employee attrition.csv")

employee <- employee %>%
  distinct() %>%
  dplyr::select(-EmployeeCount, -StandardHours, -Over18, -EmployeeNumber) %>%
  mutate(
    Attrition = ifelse(Attrition == "Yes", 1, 0),
    Attrition = as.integer(Attrition),
    across(where(is.character), as.factor)
  )

# One-hot encoding
x <- model.matrix(Attrition ~ . -1, data = employee)
y <- employee$Attrition

set.seed(123)
test_id <- createDataPartition(y, p = 0.2, list = FALSE)
x_train <- x[-test_id, ] %>% scale()
x_test <- x[test_id, ] %>% scale()
y_train <- to_categorical(y[-test_id], 2)
y_test <- to_categorical(y[test_id], 2)


model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2, activation = "softmax")


model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)


history <- model %>% fit(
  x_train, y_train,
  epochs = 100,
  batch_size = 32,
  validation_data = list(x_test, y_test)
)


plot(history)

pred_probs <- model %>% predict(x_test)
pred_class <- apply(pred_probs, 1, which.max) - 1
actual_class <- apply(y_test, 1, which.max) - 1

cm <- confusionMatrix(
  factor(pred_class, levels = c(0,1)),
  factor(actual_class, levels = c(0,1))
)
print(cm)

roc_nn <- roc(actual_class, pred_probs[,2])
plot(roc_nn, col = "blue", main = "ROC Curve - Neural Network")
auc_nn <- auc(roc_nn)
cat("Neural Network AUC:", auc_nn, "\n")






