import numpy as np

output = "\
\begin{tabular}{llll}\
$\lambda$ & $z\in[0.2,0.35)$ &$z\in[0.35,0.5)$ & $z\in[0.5,0.65)$ \\ \hline\
%s\
%s\
%s\
%s\
\end{tabular}\
"

line = "$%s$ & %.3f \pm %.3f &  %.3f \pm %.3f &  %.3f \pm %.3f \\"
labels = ["[20,30)","[30,45)","[45,60)","[60,\infty)"]

A = np.loadtxt("Y1_deltap1.txt")
var = np.loadtxt("Y1_deltap1_var.txt")
m = 0.012
m_var = 0.013**2
#A += m
#var += m_var
err = np.sqrt(var)

print A.shape, var.shape
Nz = 3
Nl = 4
for i in range(Nz):
    outline = "$%s$ "%labels[i]
    for jj in range(Nl):
        j = jj+3
        outline +="%.3f \pm %.3f "%(A[i,j]-1, err[i,j])
        if jj < 3:
            outline += "&"
        else:
            outline += "\\"
    print outline
