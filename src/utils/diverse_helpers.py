import matplotlib.text as mtext

"""
diverse helper functions and classes
"""

def secure_division_zero(n, d):
    """Avoid ZeroDivision Error
    
    Parameters
    ----------
    n : int or float
    d : int or float
    
    Returns
    -------
    int or float
    """
    return n / d if d else 0

      
class LegendTitle(object):
    """Set subtitles in matplotlib legend
    
    based on https://stackoverflow.com/questions/38463369/subtitles-within-matplotlib-legend

    Parameters
    ----------
    text_props : dict or None, optional
        defaults to None

    Attributes
    ----------
    _text_props : dict
    """
    def __init__(self, text_props=None):
        self._text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        """Return the artist that this HandlerBase generates for the given
        original artist/handle.

        Parameters
        ----------
        legend : `~matplotlib.legend.Legend`
            The legend for which these legend artists are being created.
        orig_handle : :class:`matplotlib.artist.Artist` or similar
            The object for which these legend artists are being created.
        fontsize : int
            The fontsize in pixels. The artists being created should
            be scaled according to the given fontsize.
        handlebox : `matplotlib.offsetbox.OffsetBox`
            The box which has been created to hold this legend entry's
            artists. Artists created in the `legend_artist` method must
            be added to this handlebox inside this method.
        """
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, **self._text_props)
        handlebox.add_artist(title)
        return title