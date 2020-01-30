# -*- coding: utf-8 -*-
"""
마이크로 체스와 관계없는 기타 코드 모음
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'

import logging

logger = logging.getLogger(__name__)


def ascii_plot(xs, ys, title=None, print_out=False):
    """
    gnuplot을 이용해서 ascii 문자로 꺽은선 그래프를 그려줌
    GNU plot을 별도로 설치해야 함
    - Windows (C:/Program Files/gnuplot/bin/gnuplot.exe)
    - Linux (/usr/bin/gnuplot)
    설치가 되어있지 않으면 경고 메시지만 출력함

    >>> from scripts.utils import ascii_plot
    >>> fig = ascii_plot([0, 1, 2], [0.1, 0.2, 0.4])
    >>> print(fig)

    .. code-block:: none

          0.4 +------------------------------------------------------------------------------+
              |                   +                   +                  +               *** |
              |                                                                        **    |
              |                                                                     ***      |
         0.35 |-+                                                                ***       +-|
              |                                                               ***            |
              |                                                            ***               |
          0.3 |-+                                                        **                +-|
              |                                                       ***                    |
              |                                                    ***                       |
              |                                                 ***                          |
         0.25 |-+                                            ***                           +-|
              |                                            **                                |
              |                                         ***                                  |
          0.2 |-+                                   **A*                                   +-|
              |                               ******                                         |
              |                         ******                                               |
         0.15 |-+                 ******                                                   +-|
              |              *****                                                           |
              |        ******                                                                |
              |  ******           +                   +                  +                   |
          0.1 +------------------------------------------------------------------------------+
              0                  0.5                  1                 1.5                  2

    :param xs: list of {int, float}, 숫자 리스트
    :param ys: list of {int, float}, 숫자 리스트
    :param title: str, 그래프 위에 표시할 문자열
    :param print_out: bool, True일 때는 반환하는 것과 별개로 그래프를 print 함
    :returns: str, ascii 문자로 만든 그래프
    """
    import subprocess
    import platform
    if platform.system().lower() == 'windows':
        path = 'C:/Program Files/gnuplot/bin/gnuplot.exe'
    else:
        path = "/usr/bin/gnuplot"

    try:
        gnuplot = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        input_string = "set term dumb 90 25\n"
        input_string += "set key off\n"

        if not hasattr(xs[0], '__len__'):
            xs_list = [xs]
        else:
            xs_list = xs

        if not hasattr(ys[0], '__len__'):
            ys_list = [ys]
        else:
            ys_list = ys

        input_string += "plot "
        input_string += ', '.join(["'-' using 1:2 title 'Line1' with linespoints" for ys in ys_list])
        input_string += '\n'

        for xs, ys in zip(xs_list, ys_list):
            for i, j in zip(xs, ys):
                input_string += "%f %f\n" % (i, j)
            input_string += "e\n"

        output_string, error_msg = gnuplot.communicate(input_string)

        if title is not None:
            title = '** {} **\n'.format(title.title())
            output_string = title + output_string

        if print_out:
            print(output_string)

        return output_string
    except FileNotFoundError:
        logger.error("Can't find gnuplot")
        return ''

