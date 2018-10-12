using CSV
using Statistics
using StatsBase

function load_data(labels::Vector = [0,1], frac_val::Real = 0.1)

    # read training data
    d_train = Matrix(CSV.read(pwd()*"/ds4/pima-trn_edit.csv"; allowmissing=:auto,header=0 ))
    x_train = d_train[:,1:8]
    y_train = convert(Array{Int64,1},d_train[:,end])

    # read test data
    d_test = Matrix(CSV.read(pwd()*"/ds4/pima-val_edit.csv"; allowmissing=:auto,header=0 ))
    x_test = d_test[:,1:8]
    y_test = convert(Array{Int64,1},d_test[:,end])

    # normalize data
    x_train = x_train ./ std(x_train, dims = 1)
    x_test = x_test ./ std(x_test, dims = 1)

    # split training data to training data and val data
    nbr_obs_train = size(x_train,1)
    nbr_val = Int(floor(nbr_obs_train*frac_val))

    idx_val = sample(1:nbr_obs_train, nbr_val)
    idx_train = setdiff(1:nbr_obs_train, idx_val)

    x_val = x_train[idx_val,:]
    x_train = x_train[idx_train,:]

    y_val = y_train[idx_val]
    y_train = y_train[idx_train]

    # transpose data
    x_train = Array(x_train')
    x_test = Array(x_test')
    x_val = Array(x_val')

    # set labels for classes
    y_train[findall(y_train .== 1)] .= labels[2]
    y_train[findall(y_train .== 0)] .= labels[1]

    y_val[findall(y_val .== 1)] .= labels[2]
    y_val[findall(y_val .== 0)] .= labels[1]

    y_test[findall(y_test .== 1)] .= labels[2]
    y_test[findall(y_test .== 0)] .= labels[1]

    return x_train,y_train,x_val,y_val,x_test,y_test
end


function class_logistic(y)

    class = zeros(Int, length(y))

    for i in 1:length(y)
        if y[i] > 0.5
            class[i] = 1
        end
    end

    return class

end


function class_nll(y,labels)

    class = zeros(Int, size(y,2))

    for i = 1:size(y,2)
        if findmax(y[:,i])[2]  == 1
            class[i] = labels[1]
        else
            class[i] = labels[2]
        end
    end

    return class

end


function classification_results(y_hat, y, classes)

    idx_pos = findall(y .== classes[1])
    idx_neg = findall(y .== classes[2])

    TP = sum(y_hat[idx_pos] .== classes[1])
    TN = sum(y_hat[idx_neg] .== classes[2])

    sens = TP/length(idx_pos)
    spec = TN/length(idx_neg)
    acc = (TP+TN)/length(y)

    return acc,sens,spec

end
