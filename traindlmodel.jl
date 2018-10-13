# train DL and clac predictions

include(pwd()*"/dlmodel.jl")

Random.seed!(1234) # set random numbers

# training
dropout_percentage = 0.1
lambda = [0.01, 0.01, 0.01]
@time loss_train, loss_val = train(5000, dropout_percentage, lambda)

# calc predictions
y_pred_train_proc = predict(w, x_train, 0)
y_pred_val_proc = predict(w, x_val, 0)
y_pred_proc = predict(w, x_test, 0)

y_pred_train = class_nll(y_pred_train_proc,[1,2])
y_pred_val = class_nll(y_pred_val_proc,[1,2])
y_pred_test = class_nll(y_pred_proc,[1,2])

# loss
loss_training = loss(w,x_train, y_train, 0, lambda)/length(y_train)
loss_val = loss(w,x_val, y_val, 0, lambda)/length(y_val)
loss_test = loss(w,x_test, y_test, 0, lambda)/length(y_test)

# classification results
class_res_training = classification_results(y_pred_train, y_train, [1,2])
class_res_val = classification_results(y_pred_val, y_val, [1,2])
class_res_test = classification_results(y_pred_test, y_test, [1,2])

# loss as function of epoch
PyPlot.figure()
PyPlot.plot(loss_train, "b")

PyPlot.figure()
PyPlot.plot(loss_val, "r")
