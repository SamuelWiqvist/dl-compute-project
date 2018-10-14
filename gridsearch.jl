# find hyperparameter values using gird search

include(pwd()*"/dlmodel.jl")

p = collect(0:0.2:1)

lambda_1 = [0; collect(0.00001:0.02:0.1)]
lambda_2 = [0; collect(0.00001:0.02:0.1)]
lambda_3 = [0; collect(0.00001:0.02:0.1)]

model_fit_regularization = zeros(length(p),length(lambda_1),length(lambda_2),length(lambda_3),3+3+1+1)

# find hyperparameters using gridsearch

Random.seed!(1234) # set random numbers

@time for i in 1:length(p)
    @printf "Percentage done %.2f\n" (i-1)/length(p)
    for j in 1:length(lambda_1)
        for k in 1:length(lambda_2)
            for l in 1:length(lambda_3)

                w = Any[ xavier(n_hidden_1,n_input), zeros(n_hidden_1,1),
                         xavier(n_hidden_2,n_hidden_1), zeros(n_hidden_2,1),
                         xavier(n_out,n_hidden_2), zeros(n_out,1)]

                optim = optimizers(w, Adam)

                dropout_percentage = p[i]
                lambda = [lambda_1[j], lambda_2[k], lambda_3[l]]
                loss_train, loss_val = train(5000, dropout_percentage, lambda)

                # calc predictions
                y_pred_train_proc = predict(w, x_train, 0)
                y_pred_val_proc = predict(w, x_val, 0)

                y_pred_train = class_nll(y_pred_train_proc,[1,2])
                y_pred_val = class_nll(y_pred_val_proc,[1,2])

                # loss
                model_fit_regularization[i,j,k,l,end-1] = loss(w,x_train, y_train, 0, lambda)/length(y_train)
                model_fit_regularization[i,j,k,l,end] = loss(w,x_val, y_val, 0, lambda)/length(y_val)

                # classification results
                model_fit_regularization[i,j,k,l,1:3] = classification_results(y_pred_train, y_train, [1,2])
                model_fit_regularization[i,j,k,l,4:6] = classification_results(y_pred_val, y_val, [1,2])

            end
        end
    end
end

# results without regularization
training_loss_no_reg = model_fit_regularization[1,1,1,1,end-1]
test_loss_no_reg = model_fit_regularization[1,1,1,1,end]
training_acc_no_reg = model_fit_regularization[1,1,1,1,1]
test_acc_no_reg = model_fit_regularization[1,1,1,1,4]

# find best hyperparameter settings 
(~, idx_best_training_loss) = findmin(model_fit_regularization[:,:,:,:,end-1])
(~, idx_best_test_loss) = findmin(model_fit_regularization[:,:,:,:,end])

(~, idx_best_training_acc) = findmax(model_fit_regularization[:,:,:,:,1])
(~, idx_best_test_acc) = findmax(model_fit_regularization[:,:,:,:,4])
