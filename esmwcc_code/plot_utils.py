import  numpy as np
JITTER = 0.01


def jitter(points):
    x, y = [p[0] for p in points], [p[1] for p in points]
    x_jitter = np.random.normal(x, JITTER, size=len(x))
    y_jitter = np.random.normal(y, JITTER, size=len(y))

    return x_jitter, y_jitter


def clean(points):
    for i in range(len(points)):
        if points[i][0] == -10000 or points[i][1] == -10000:
            print("Found point without valid value. Removing from plot.")
            points[i] = None
    points = [p for p in points if p is not None]
    return points


def soft_logit(x):
    x = np.clip(x, 1e-3, 1-1e-3)
    return np.log(x / (1 - x))


def get_dimensions(task='paraphrase'):
    #fig, ax = plt.subplots(figsize=(4, 3)) is for 0.5\textwidth
    # it will be included in a pdf, using  \begin{subfigure}[b]{0.32\textwidth} and \includegraphics[width=\textwidth]{plot.pdf}
    #return 3, 2.5
    return 3, 2.5


def colors_matplotlib(task='paraphrase') -> dict:
    raise NotImplementedError


