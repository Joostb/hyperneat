# hyperneat
A repo for the NC course

# Plan
We want to implement HyperNeat.
According to the paper [1], we will have to implement NEAT[4,5] first, or use an existing NEAT implementation.


# Some relevant papers:


 1. The original HyperNeat Paper. It is really long (30 pages), the first 13 pages are about the theory behind Hyperneat, after that it's just about experiments. <br>
   http://eplex.cs.ucf.edu/papers/stanley_alife09.pdf

 2. Hyperneat applied on Atari games, also long (18 pages) and not really indepth on the implementation, but might be nice for trainining tips. <br>
   http://www.cs.utexas.edu/users/pstone/Papers/bib2html-links/TCIAIG13-mhauskn.pdf
   
 3. A paper that uses hyperneat to classify MNIST digits with horrible results 🙃. <br>
    https://arxiv.org/pdf/1312.5355.pdf
    
 4. Original NEAT paper: <br>
    http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
    
 5. Followup to the NEAT paper (quote: "Each of the 33 evolution runs took between 5 and 10 days on a 1GHz Pentium III processor,
depending on the progress of evolution and sizes of the networks involved"): <br>
    http://nn.cs.utexas.edu/downloads/papers/stanley.jair04.pdf

# reference implementations

C++ (original): https://github.com/mhauskn/HyperNEAT
Java: https://github.com/OliverColeman/ahni

C++, python bindings (according to the hyperneat page, it is heavily documented): https://github.com/peter-ch/MultiNEAT/tree/master/src
