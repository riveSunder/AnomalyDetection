import numpy.random as npr
from autograd import numpy as np
from autograd import grad

import sklearn
import sklearn.covariance
import sklearn.svm
import sklearn.ensemble

import matplotlib.pyplot as plt

def make_fourier_function(my_seed=42, terms=4):

    np.random.seed(my_seed)

    def my_function(x):
    
        y = np.zeros(x.shape)
        # dc (average value) term
        y += np.random.rand()

        for ii in range(terms):
            y += np.random.rand() * np.sin(ii*x)

        return y
        
    return my_function

relu = lambda x: x * (x > 0.0)

def mlp_forward(x, w, act=np.tanh):
    
    for weights in w[:-1]:
        x = act(np.matmul(x, weights))

    return np.matmul(x, w[-1])

def make_mlp_function(w):
    
    def my_mlp_function(x):
        x = mlp_forward(x, w)
        return x

    return my_mlp_function

def make_mlp_loss_function(loss_fn=lambda y, y_: np.mean((y - y_)**2)):

    def my_mlp_loss_function(x, y, w):
        pred = mlp_forward(x, w)
        loss = loss_fn(y, pred)

        return loss

    return my_mlp_loss_function

def make_mlp_loss_grad_functions(loss_fn=lambda y, y_: np.mean((y-y_)**2)):
    
    my_mlp_loss_function = make_mlp_loss_function(loss_fn)
    my_grad_function = grad(my_mlp_loss_function, argnum=2)

    return my_mlp_loss_function, my_grad_function

def update_w(w, w_grad, lr=1e-3):

    for ii in range(len(w)):
        
        w[ii] = w[ii] - lr * w_grad[ii]

    return w

def train_mlp(x, y, w, epochs=100, batch_size=None):

    batch_size = x.shape[0] if batch_size is None else batch_size
    display_every = epochs // 10
    loss_fn, grad_fn = make_mlp_loss_grad_functions()

    for epoch in range(epochs):

        if epoch % display_every == 0:
            loss = loss_fn(x, y, w) 

            print(f"loss at epoch {epoch}: {loss:.3e}")

        for batch_index in range(0, x.shape[0], batch_size):

            batch_x = x[batch_index:batch_index + batch_size]
            batch_y = y[batch_index:batch_index + batch_size]

            w_grad = grad_fn(batch_x, batch_y, w)

            w = update_w(w, w_grad)

    return w

def make_anomaly_fig(val_y, outliers, in_predict_0, out_predict_0 ):

    fig = plt.figure(figsize=(8,8))

    # true inliers
    plt.plot(val_y[in_predict_0 > 0.0][:,0],\
        val_y[in_predict_0 > 0.0][:,1],\
        "o", markerfacecolor="g", markeredgecolor="g", alpha=0.5, \
        markeredgewidth=3, markersize=9, label="true inliers")
    # false outliers
    plt.plot(val_y[in_predict_0 < 0.0][:,0],\
        val_y[in_predict_0 < 0.0][:,1],\
        "o", markerfacecolor="g", markeredgecolor="r", alpha=0.5, \
        markeredgewidth=3, markersize=9, label="false outliers")

    # false inliers
    plt.plot(outliers[out_predict_0 > 0.0][:,0],\
        outliers[out_predict_0 > 0.0][:,1],\
        "o", markerfacecolor="r", markeredgecolor="g", alpha=0.5, \
        markeredgewidth=3, markersize=9, label="false inliers")
    # true outliers
    plt.plot(outliers[out_predict_0 < 0.0][:,0],\
        outliers[out_predict_0 < 0.0][:,1],\
        "o", markerfacecolor="r", markeredgecolor="r", alpha=0.5,\
        markeredgewidth=3, markersize=9, label="true_outliers")

    return fig

def mlp_predict_outliers(in_x, predict_x, w, std_dev=2):

    sample_loss_fn = make_mlp_loss_function(loss_fn = lambda y, y_: np.mean((y - y_)**2, axis=1))

    in_loss = sample_loss_fn(in_x, in_x, w)
    in_std_dev = np.std(in_loss)

    predict_loss = sample_loss_fn(predict_x, predict_x, w)

    predict = 0.0 * predict_loss

    predict[predict_loss > (std_dev * in_std_dev)] = -1.0
    predict[predict_loss <= (std_dev * in_std_dev)] = 1.0

    return predict

class MLPEstimator():

    def __init__(self):

        self.w_fit = [npr.randn(2,32), npr.randn(32,2)]

    def fit(self, train_y, epochs=1000, batch_size=None):

        if batch_size is None:
            batch_size = train_y.shape[0]

        self.w_fit = train_mlp(train_y, train_y, self.w_fit, \
                epochs=1000, batch_size=batch_size)

        self.train_y = train_y

    def predict(self, my_data, sd=5):

        return mlp_predict_outliers(self.train_y, my_data, self.w_fit, std_dev=sd)


if __name__ == "__main__":

    for seed in [1, 3, 5, 7, 11, 13, 17, 19]:
        npr.seed(seed)
        w_inlier = [npr.randn(2,32), npr.randn(32,2)]
        w_outlier = [npr.randn(2,32), npr.randn(32,2)]

        mlp_inlier = make_mlp_function(w_inlier)
        mlp_outlier = make_mlp_function(w_outlier)

        display_x = np.arange(-1.0, 1.0, 0.01).reshape(-1,1)

        train_x = npr.randn(4096,2)  
        val_x = npr.randn(128,2)  

        train_y = mlp_inlier(train_x)
        val_y = mlp_inlier(val_x)
        outliers = mlp_outlier(val_x[:64,:])

        estimator_0 = sklearn.covariance.EllipticEnvelope()
        estimator_1 = sklearn.ensemble.IsolationForest()
        estimator_2 = sklearn.svm.OneClassSVM()
        estimator_3 = MLPEstimator()

        estimator_0.fit(train_y)
        estimator_1.fit(train_y)
        estimator_2.fit(train_y)
        estimator_3.fit(train_y)

        in_predict_0 = estimator_0.predict(val_y)
        in_predict_1 = estimator_1.predict(val_y)
        in_predict_2 = estimator_2.predict(val_y)
        in_predict_3 = estimator_3.predict(val_y)

        out_predict_0 = estimator_0.predict(outliers)
        out_predict_1 = estimator_1.predict(outliers)
        out_predict_2 = estimator_2.predict(outliers)
        out_predict_3 = estimator_3.predict(outliers)

        print(seed)

        fig0 = make_anomaly_fig(val_y, outliers, in_predict_0, out_predict_0)
        plt.title("covariance.EllipticEnvelope", fontsize=32)
        correct = np.sum(in_predict_0 > 0.0) + np.sum(out_predict_0 < 0.0)
        accuracy = correct / (val_y.shape[0] + outliers.shape[0])
        plt.xlabel(f"accuracy: {accuracy:.3f}", fontsize=16)
        plt.legend()
        plt.savefig(f"./assets/acc_{int(100*accuracy)}_seed{seed}sk_covariance_.png")

        fig1 = make_anomaly_fig(val_y, outliers, in_predict_1, out_predict_1)
        plt.title("IsolationForest", fontsize=32)
        correct = np.sum(in_predict_1 > 0.0) + np.sum(out_predict_1 < 0.0)
        accuracy = correct / (val_y.shape[0] + outliers.shape[0])
        plt.xlabel(f"accuracy: {accuracy:.3f}", fontsize=16)
        plt.legend()
        plt.savefig(f"./assets/acc_{int(100*accuracy)}_seed{seed}sk_isolation_forest_.png")

        fig2 = make_anomaly_fig(val_y, outliers, in_predict_2, out_predict_2)
        plt.title("OneClassSVM", fontsize=32)
        correct = np.sum(in_predict_2 > 0.0) + np.sum(out_predict_2 < 0.0)
        accuracy = correct / (val_y.shape[0] + outliers.shape[0])
        plt.xlabel(f"accuracy: {accuracy:.3f}", fontsize=16)
        plt.legend()
        plt.savefig(f"./assets/acc_{int(100*accuracy)}_seed{seed}sk_svm_.png")


        fig3 = make_anomaly_fig(val_y, outliers, in_predict_3, out_predict_3)
        plt.title(f"Autoencoder Loss > 5 $\sigma$", fontsize=32)
        correct = np.sum(in_predict_3 > 0.0) + np.sum(out_predict_3 < 0.0)
        accuracy = correct / (train_y.shape[0] + outliers.shape[0])
        plt.xlabel(f"accuracy: {accuracy:.3f}", fontsize=16)
        plt.legend()
        plt.savefig(f"./assets/acc_{int(100*accuracy)}_seed{seed}_mlp.png")
