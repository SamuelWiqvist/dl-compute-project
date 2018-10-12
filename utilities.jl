using CSV
using Statistics
using StatsBase

function load_data(frac_val::Real = 0.1)

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

    return x_train,y_train,x_val,y_val,x_test,y_test
end


function class(y)

    class = zeros(Int, size(y,2))

    for i in 1:size(y,2)
        if y[1,i] > 0.5
            class[i] = 1
        end
    end

    return class

end
