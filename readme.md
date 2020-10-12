# Millimeter Wave Channel Modeling via Generative Neural Networks

* Sundeep Rangan, William Xia, Marco Mezzavilla, Giuseppe Loianno (NYU)
* Giovanni Geraci, Angel Lozano (UPF, Barcelona)
* Vasilii Semkin (VTT, Finland)

Statistical channel models are instrumental to design and evaluate wireless 
communication systems. In the millimeter wave bands, such models become 
acutely challenging: they must capture the delay, directions, and path gains, 
for each link and with high resolution.  Data-driven machine-learning methods 
provides an attractive methodology that entails minimal assumptions and can 
naturally capture intricate probabilistic relationships in complex 
environments.  This repository is currently in progress and will eventually 
provide tools for:

*  Parsing large quantities of ray tracing data for generating ML training 
and test datasets. 
*  Learning generative neural network for statistical models of the data.
*  Access to pre-trained models including UAV ground-to-air channels at 28 GHz.
*  Performing simple network simulation studies with these models. 

The work is partly based on

* Xia, W., Rangan, S., Mezzavilla, M., Lozano, A., Geraci, G., Semkin, V., & Loianno, G. (2020). Millimeter Wave Channel Modeling via Generative Neural Networks. arXiv preprint arXiv:2008.11006.

   
## Acknowledgments
W.  Xia,  M.  Mezzavilla  and  S.  Rangan  were  supportedby  NSF  grants  1302336,  1564142,  1547332,  and  1824434,NIST, SRC, and the industrial affiliates of NYU WIRELESS.A.  Lozano  and  G.  Geraci  were  supported  by  the  ERC  grant694974,  by  MINECO’s  Project  RTI2018-101040,  and  by  the Junior Leader Program from "la Caixa" Banking Foundation.

All authors are also grateful for the assistance from Remcom that provided
the Wireless Insite tool to generate the data.
