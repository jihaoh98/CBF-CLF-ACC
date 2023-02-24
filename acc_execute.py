import numpy as np
import dynamics
import cbf_clf
import matplotlib.pyplot as plt

class ACC_Execute:
    def __init__(self) -> None:
        self.parameter = {
                    'f0': .1,
                    'f1': 5,
                    'f2': .25,
                    'v0': 10,
                    'm': 1650,
                    'T': 1.8,
                    'cd': .3,
                    'vd': 24,
                    'G': 9.8,
                    'udim': 1
                }
        
        self.vehicle = dynamics.AdaptiveCruiseControl(self.parameter)
        self.Qp_option = cbf_clf.CbfClfQpOptions()

        # set the option
        self.Qp_option.set_option('u_max', np.array([self.parameter['cd'] * self.parameter['G'] * self.parameter['m']]))
        self.Qp_option.set_option('u_min', np.array([-self.parameter['cd'] * self.parameter['G'] * self.parameter['m']]))
        self.Qp_option.set_option('clf_lambda', 5.0)
        self.Qp_option.set_option('cbf_gamma', 5.0)
        self.Qp_option.set_option('weight_input', np.array([2 / self.parameter['m'] **2]))
        self.Qp_option.set_option('weight_slack', 2e-2)

        self.qp = cbf_clf.CBF_CLF_Qp(self.vehicle, self.Qp_option)

        # simulation parameter
        # maximum simuation time
        self.T = 20

        # step time
        self.dt = .02
        self.time_steps = int(np.ceil(self.T / self.dt))

        self.x0 = np.array([0, 20, 100])
        self.x = self.x0

        # storage
        self.xt = np.zeros((3, self.time_steps))
        self.ut = np.zeros((self.parameter['udim'], self.time_steps))
        self.slackt = np.zeros((1, self.time_steps))
        self.clf_t = np.zeros((1, self.time_steps))
        self.cbf_t = np.zeros((1, self.time_steps))

    def qp_solve(self):
        """ solve the qp """
        for t in range(self.time_steps):
            if t % 100 == 0:
                print(f't = {t}')
    
            u_ref = self.vehicle.get_reference_control(self.x)
            u, delta, cbf, clf, feas = self.qp.cbf_clf_qp(self.x, u_ref)

            if not feas:
                print('This problem is infeasible!')
                break
            else:
                pass
            
            self.xt[:, t] = np.copy(self.x)
            self.ut[:, t] = u
            self.slackt[:, t] = delta
            self.cbf_t[:, t] = cbf
            self.clf_t[:, t] = clf

            self.x = self.vehicle.next_state(self.x, u, self.dt)

        print('Finish the solve of qp!')

    def show_velocity(self):
        t = np.arange(0, self.T, self.dt)
        fig = plt.figure(figsize=[16, 9])
        plt.grid()

        plt.plot(t, self.xt[1, :], linewidth=3, color='magenta')
        plt.plot(t, self.parameter['vd'] * np.ones(t.shape[0]), 'k--')
        plt.title('State - Velocity')
        plt.ylabel('v')

        plt.show()
        # plt.savefig('velocity.png', format='png', dpi=300)
        plt.close(fig)

    def show_realtive_distance(self):
        t = np.arange(0, self.T, self.dt)
        fig = plt.figure(figsize=[16, 9])
        plt.grid()

        plt.plot(t, self.xt[2, :], linewidth=3, color='black')
        plt.ylim(0, 100)
        plt.title('State - Relative distance')
        plt.ylabel('z')

        plt.show()
        # plt.savefig('relative_distance.png', format='png', dpi=300)
        plt.close(fig)

    def show_slack(self):
        t = np.arange(0, self.T, self.dt)
        fig = plt.figure(figsize=[16, 9])
        plt.grid()
        
        plt.plot(t, self.slackt[0], linewidth=3, color='orange')
        plt.title('Slack')
        plt.ylabel('delta')

        plt.show()
        # plt.savefig('slack.png', format='png', dpi=300)
        plt.close(fig)

    def show_cbf(self):
        t = np.arange(0, self.T, self.dt)
        fig = plt.figure(figsize=[16, 9])
        plt.grid() 

        plt.plot(t, self.cbf_t[0], linewidth=3, color='red')
        plt.title('cbf')
        plt.ylabel('h(x)')

        plt.show()
        # plt.savefig('cbf.png', format='png', dpi=300)
        plt.close(fig)

    def show_clf(self):
        t = np.arange(0, self.T, self.dt)
        fig = plt.figure(figsize=[16, 9])
        plt.grid() 

        plt.plot(t, self.clf_t[0], linewidth=3, color='cyan')
        plt.title('clf')
        plt.ylabel('V(x)')

        plt.show()
        # plt.savefig('clf.png', format='png', dpi=300)
        plt.close(fig)

    def show_control(self):
        u_max = self.parameter['cd'] * self.parameter['G'] * self.parameter['m']
        t = np.arange(0, self.T, self.dt)
        fig = plt.figure(figsize=[16, 9])
        plt.grid() 

        plt.plot(t, self.ut[0], linewidth=3, color='dodgerblue')
        plt.plot(t, u_max * np.ones(t.shape[0]), 'k--')
        plt.plot(t, -u_max * np.ones(t.shape[0]), 'k--')

        plt.title('control')
        plt.ylabel('u')

        plt.show()
        # plt.savefig('control.png', format='png', dpi=300)
        plt.close(fig)

    def storage_data(self):
        np.savez('process_data', x=self.xt, slack=self.slackt, cbf=self.cbf_t, clf=self.clf_t, u=self.ut)

if __name__ == '__main__':
    test_target = ACC_Execute()
    test_target.qp_solve()
    
    test_target.show_velocity()
    test_target.show_realtive_distance()
    test_target.show_slack()
    test_target.show_cbf()
    test_target.show_clf()
    test_target.show_control()

    test_target.storage_data()
    
