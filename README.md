# Deep Learning of Near Field Beam Focusing in Terahertz Wideband Massive MIMO Systems
This is the simulation codes related to the following article: Y. Zhang and A. Alkhateeb, "[Deep Learning of Near Field Beam Focusing in Terahertz Wideband Massive MIMO Systems](https://ieeexplore.ieee.org/document/10004962)," in IEEE Wireless Communications Letters, doi: 10.1109/LWC.2022.3233566.

# Abstract of the Article
Employing large antenna arrays and utilizing large bandwidth have the potential of bringing very high data rates to future wireless communication systems. However, this brings the system into the near-field regime and also makes the conventional transceiver architectures suffer from the wideband effects. To address these problems, in this paper, we propose a low-complexity frequency-aware beamforming solution that is designed for hybrid time-delay and phase-shifter based RF architectures. To reduce the complexity, the joint design problem of the time delays and phase shifts is decomposed into two subproblems, where a signal model inspired online learning framework is proposed to learn the shifts of the quantized analog phase shifters, and a low-complexity geometry-assisted method is leveraged to configure the delay settings of the time-delay units. Simulation results highlight the efficacy of the proposed solution in achieving robust performance across a wide frequency range for large antenna array systems.

# How to generate this codebook beam patterns figure?
1. Download all the files of this repository.
2. Run `main.py` in `critic_net_training` directory.
3. After it is finished, there will be a file named `critic_params_trsize_2000_epoch_500_3bit.mat` that will be used in the next step.
4. Run `main.py` in `analog_beam_learning` directory.
5. After it is finished, run `read_beams.py` in the same directory.
6. Copy the generated file, i.e., `ULA_PS_only.mat` to the `td_searching` directory.
7. Run `NFWB_BF_TTD_PS_hybrid_low_complexity_search_algorithm.m` in Matlab, which will generate the figure shown below.

![Figure](https://github.com/YuZhang-GitHub/NFWB_BF/blob/main/N_16.png)

If you have any problems with generating the figure, please contact [Yu Zhang](https://www.linkedin.com/in/yu-zhang-391275181/).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you in any way use this code for research that results in publications, please cite our original article:
> Y. Zhang and A. Alkhateeb, "[Deep Learning of Near Field Beam Focusing in Terahertz Wideband Massive MIMO Systems](https://ieeexplore.ieee.org/document/10004962)," in IEEE Wireless Communications Letters, doi: 10.1109/LWC.2022.3233566.
