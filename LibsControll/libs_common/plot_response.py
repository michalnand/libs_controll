import matplotlib.pyplot as plt


def plot_output(t_trajectory, u_trajectory, y_trajectory, u_names, y_names, output_file_name):

    t_trajectory = t_trajectory.detach().to("cpu").numpy()
    u_trajectory = u_trajectory.detach().to("cpu").numpy()
    y_trajectory = y_trajectory.detach().to("cpu").numpy()


    u_mean  = u_trajectory.mean(axis=1)
    u_min   = u_trajectory.min(axis=1)
    u_max   = u_trajectory.max(axis=1) 

    y_mean  = y_trajectory.mean(axis=1)
    y_min   = y_trajectory.min(axis=1)
    y_max   = y_trajectory.max(axis=1)

   
    plt.clf()
    fig, axs = plt.subplots(len(u_names) + len(y_names), 1, figsize=(7,9))

    for i in range(len(u_names)):
        axs[i].fill_between(t_trajectory, u_min[:, i], u_max[:, i], facecolor='red', alpha=0.3)
        axs[i].plot(t_trajectory, u_mean[:, i], color='red')
        axs[i].set_xlabel("time [s]")
        axs[i].set_ylabel(u_names[i])
        axs[i].grid(True)

    for i in range(len(y_names)):
        axs[i + len(u_names)].fill_between(t_trajectory, y_min[:, i], y_max[:, i], facecolor='deepskyblue', alpha=0.3)
        axs[i + len(u_names)].plot(t_trajectory, y_mean[:, i], color='deepskyblue')
        axs[i + len(u_names)].set_xlabel("time [s]")
        axs[i + len(u_names)].set_ylabel(y_names[i])
        axs[i + len(u_names)].grid(True)

    fig.tight_layout()
    plt.savefig(output_file_name, dpi = 300)
    plt.close()


def plot_controll_output(t_trajectory, u_trajectory, y_req_trajectory, y_test_trajectory, u_names, y_names, output_file_name):

    t_trajectory        = t_trajectory.detach().to("cpu").numpy()
    u_trajectory        = u_trajectory.detach().to("cpu").numpy()
    y_req_trajectory    = y_req_trajectory.detach().to("cpu").numpy()
    y_test_trajectory   = y_test_trajectory.detach().to("cpu").numpy()



    plt.clf()
    fig, axs = plt.subplots(len(u_names) + len(y_names), 1, figsize=(7,10))
    
    idx = 0
    for i in range(len(u_names)):
        if u_names[i] is not None:
            axs[i].plot(t_trajectory, u_trajectory[:, i], color='red')
            axs[i].set_xlabel("time [s]")
            axs[i].set_ylabel(u_names[i])
            axs[i].grid(True)

        idx+= 1


    for i in range(len(y_names)):
        if y_names[i] is not None:
            axs[idx].plot(t_trajectory, y_req_trajectory[:, i], color='red', label="required output")
            axs[idx].plot(t_trajectory, y_test_trajectory[:, i], color='deepskyblue', label="plant output")
            
            axs[idx].set_xlabel("time [s]")
            axs[idx].set_ylabel(y_names[i])
            axs[idx].grid(True)

            axs[idx].legend(loc='upper left')
            
        idx+= 1

    fig.tight_layout()
    plt.savefig(output_file_name, dpi = 300)
    plt.close()
