from netCDF4 import Dataset as open_ncfile
import netCDF4 as nc
import numpy as np
import os.path
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Panel, Tabs, CustomJS, RangeTool, CheckboxButtonGroup, Range1d, PreText, \
    ColorBar, FixedTicker, LinearColorMapper, PrintfTickFormatter, WheelZoomTool
from bokeh.plotting import figure, show, output_file
import bokeh.palettes as palettes
import pandas as pd
import matplotlib.pyplot as plt

RAW_DATA_FOLDER = "data/"
PROCESSED_DATA_FOLDER = "processed_data/"
IMAGE_FOLDER = "images/"


def tau_map_preprocessing():
    file = RAW_DATA_FOLDER + "tau_map_2012-12.nc"
    nc = open_ncfile(file)

    aod_var = nc.variables['tau'][:, :]
    aod = np.ndarray((1, len(aod_var[:, 0]), len(aod_var[0, :])))
    aod[0] = aod_var
    wavelength = [550]
    time_units = 'days'
    time = nc.variables['month'][:]
    latitude = nc.variables['lat'][:]

    aod_obj = AodData(aod, latitude, time, time_units, wavelength)
    aod_obj.save_as_nc(PROCESSED_DATA_FOLDER + 'processed_tau_map.nc')


def ammann_preprocessing():

    file = RAW_DATA_FOLDER + "ammann2003b_volcanics.nc"
    nc = open_ncfile(file)

    aod_var = nc.variables['TAUSTR'][:, :]
    aod = np.ndarray((1, len(aod_var[:, 0]), len(aod_var[0, :])))
    aod[0] = aod_var
    wavelength = [550]
    time = nc.variables['time'][:]

    years = []
    # Convert yyyymm format to year format using center of each month
    for index, element in enumerate(time):
        fractional_year = float(str(element)[0:4]) + (float(str(element)[4:6])-0.5)/12
        fractional_year = round(fractional_year, 3)
        years.append(fractional_year)
    time = years
    time_units = 'years'

    latitude = nc.variables['lat'][:]

    aod_obj = AodData(aod, latitude, time, time_units, wavelength)
    aod_obj.save_as_nc(PROCESSED_DATA_FOLDER + 'processed_ammann.nc')


def evolv2k_preprocessing():
    file = RAW_DATA_FOLDER + "eVolv2k_v3_EVA_AOD_-500_1900_1.nc"
    nc = open_ncfile(file)
    aod_var = nc.variables['aod550'][:, :]
    aod = np.ndarray((1, len(aod_var[:, 0]), len(aod_var[0, :])))
    aod[0] = aod_var
    wavelength = [550]
    time_units = 'years'
    time = nc.variables['time']
    latitude = nc.variables['lat']
    aod_obj = AodData(aod, latitude, time, time_units, wavelength)
    aod_obj.save_as_nc(PROCESSED_DATA_FOLDER + 'processed_eVolv2k.nc')


def volmip_preprocessing():
    file = RAW_DATA_FOLDER + "CMIP_VOLMIP550_radiation_v4_1850-2018.nc"
    nc = open_ncfile(file)

    alt = nc.variables['altitude'][:]
    wl1_earth = nc.variables['wl1_earth'][:]
    wl1_sun = nc.variables['wl1_sun'][:]
    lat = nc.variables['latitude'][:]
    ext_earth = nc.variables['ext_earth'][:, :, :, :]
    ext_sun = nc.variables['ext_sun'][:, :, :, :]
    wavelength = [550]
    time_units = 'months'
    time = nc.variables['month'][:]

    # EXT dimensions go wl1, lat, alt, time, need to rearrange
    # Want time, wl1, lat, alt
    ext_earth = np.transpose(ext_earth, (3, 0, 1, 2))
    ext_sun = np.transpose(ext_sun, (3, 0, 1, 2))

    # Want to iterate over all altitudes, in case altitude differences are not consistently spaced.
    # Build the aod arrays and initialize to zeros, so we can add to them
    aod_earth = np.zeros((len(time), len(wl1_earth), len(lat)))
    aod_sun = np.zeros((len(time), len(wl1_sun), len(lat)))
    # Extrapolate past edge of altitude, in order to get a change in altitude for first point
    last_alt = alt[0] - (alt[1] - alt[0])
    # Sum ext*d(alt) over all altitudes
    for alt_index, alt in enumerate(alt):
        aod_earth += ext_earth[:, :, :, alt_index] * (alt - last_alt)
        aod_sun += ext_sun[:, :, :, alt_index] * (alt - last_alt)
        last_alt = alt

    # Use aod_sun because it is the one at 550nm
    aod = np.transpose(aod_sun, (1, 0, 2)) # Rearrange again - AodData expects wavelength, time, latitude
    aod_obj = AodData(aod, lat, time, time_units, wavelength)
    aod_obj.save_as_nc(PROCESSED_DATA_FOLDER + 'processed_volmip.nc')


def CMIP6_preprocessing():
    file = RAW_DATA_FOLDER + "CMIP_1850_2014_extinction_550nm_strat_only_v3.nc"
    nc = open_ncfile(file)

    alt = nc.variables['altitude'][:]
    lat = nc.variables['latitude'][:]
    ext = nc.variables['ext550'][:, :, :]
    wavelength = [550]
    time_units = 'months'
    time = nc.variables['month'][:]  # months since jan 1 1850
    # NOTE the nc file data says this is in months since jan 1960, but it appears that is incorrect.

    # Add wavelength dimension to extension, only 1 because ext is at 550
    ext_tmp = np.zeros((1, len(lat), len(alt), len(time)))
    ext_tmp[0, :, :, :] = ext
    ext = ext_tmp
    # EXT dimensions go wl1, lat, alt, time, need to rearrange
    # Want time, wl1, lat, alt
    ext = np.transpose(ext, (3, 0, 1, 2))

    # Want to iterate over all altitudes, in case altitude differences are not consistently spaced.
    # Build the aod arrays and initialize to zeros, so we can add to them
    aod = np.zeros((len(time), 1, len(lat)))
    # Extrapolate past edge of altitude, in order to get a change in altitude for first point
    last_alt = alt[0] - (alt[1] - alt[0])
    # Sum ext*d(alt) over all altitudes
    for alt_index, alt in enumerate(alt):
        aod += ext[:, :, :, alt_index] * (alt - last_alt)
        last_alt = alt

    # Use aod_sun because it is the one at 550nm
    aod = np.transpose(aod, (1, 0, 2))  # Rearrange again - AodData expects wavelength, time, latitude
    aod_obj = AodData(aod, lat, time, time_units, wavelength)
    aod_obj.save_as_nc(PROCESSED_DATA_FOLDER + 'processed_cmip6.nc')


def ICI5_preprocessing():
    folder = RAW_DATA_FOLDER + 'ICI5/'
    files = [folder for x in range(4)]
    files[0] = files[0] + 'ICI5_3090N_AOD_c.txt'
    files[1] = files[1] + 'ICI5_030N_AOD_c.txt'
    files[2] = files[2] + 'ICI5_030S_AOD_c.txt'
    files[3] = files[3] + 'ICI5_3090S_AOD_c.txt'
    open_files = [open(file).readlines() for file in files]
    aod_3090_N = [float(x.split('\t')[1]) for x in open_files[0]]
    aod_030_N = [float(x.split('\t')[1]) for x in open_files[1]]
    aod_030_S = [float(x.split('\t')[1]) for x in open_files[2]]
    aod_3090_S = [float(x.split('\t')[1]) for x in open_files[3]]
    time = [float(x.split('\t')[0]) for x in open_files[0]]
    time_units = 'years'
    wavelength = [550]
    # regions are equivalent area, so can easily calculate gm aod
    gm_aod = np.zeros((1, len(time)))
    gm_aod[0, :] = [(aod_030_N[x] + aod_3090_N[x] + aod_030_S[x] + aod_3090_S[x])/4 for x in range(len(time))]
    # Need to provide a latitude vector for input, will create 18 regions (useful to have more than four for plotting).
    latitude = range(-90, 91, 1)
    # Combine aod arrays into one, with dimensions wavelength, time, latitude\
    aod = np.zeros((len(wavelength), len(time), len(latitude)))
    for index, lat in enumerate(latitude):
        if lat < -30:
            aod[0, :, index] = aod_3090_N[:]
        elif -30 <= lat <= 0:
            aod[0, :, index] = aod_030_N[:]
        elif 0 < lat <= 30:
            aod[0, :, index] = aod_030_S[:]
        else:
            aod[0, :, index] = aod_3090_S[:]

    # Combine into AOD data object
    aod_obj = AodData(aod, latitude, time, time_units, wavelength, gm_aod=gm_aod)
    aod_obj.save_as_nc(PROCESSED_DATA_FOLDER + 'processed_ICI5.nc')


def IVI2_preprocessing():
    file = RAW_DATA_FOLDER + 'IVI2LoadingLatHeight501-2000_Oct2012.txt'
    D = open(file).readlines()
    gao_lat = list(range(-85, 86, 10)) # Go to 86 so that 85 is included
    DATA = np.zeros((18000, 773))
    gao_time = np.zeros(18000)
    for index, x in enumerate(D):
        x = x.split()
        x = [float(i) for i in x]
        gao_time[index] = x[0]
        DATA[index, :] = x[1:-1]
    gao_M= np.zeros((18000, 18, 43))

    for t in range(18000):
        for lat in range(18):
            for alti in range(43):
                gao_M[t, lat, alti] = DATA[t-1, 43*(lat-1)+alti-1]

    gao_M_vsum = np.sum(gao_M, axis=2)
    Ae = 4 * np.pi * 6371 ** 2
    gao_aod = gao_M_vsum * Ae /1.5e11

    weighted_lat = np.zeros(len(gao_lat))
    gao_aod_gm = np.zeros(len(gao_aod[:, 0]))
    # Calculate the weighted latitude data
    weighted_lat[:] = np.cos(np.dot(gao_lat[:], (np.pi / 180)))
    weighted_lat[:] = weighted_lat[:] / sum(weighted_lat[:])
    # Fill the gm_aod with the properly weighed aod data
    gao_aod_gm[:] = np.inner(gao_aod[:, :], weighted_lat)

    time_units = 'years'
    wavelength = [550]

    # Give aod dimensions of wavelength, time, latitude, with only wavelength being 550
    aod = np.zeros((1, len(gao_aod[:, 0]), len(gao_aod[0, :])))
    aod[0, :, :] = gao_aod
    # Give gm aod dimensions of wavelength, time, with only wavelength being 550
    gm_aod = np.zeros((1, len(gao_aod_gm)))
    gm_aod[0] = gao_aod_gm

    aod_obj = AodData(aod, gao_lat, gao_time, time_units, wavelength, gm_aod=gm_aod)
    aod_obj.save_as_nc(PROCESSED_DATA_FOLDER + 'processed_IVI2.nc')


class AodData:

    aod = None
    gm_aod = None
    latitude = None
    time = None
    time_units = None
    wavelengths = None
    years = None
    months = None
    days = None

    def __init__(self, aod, latitude, time, time_units, wavelengths, gm_aod=None):
        self.aod = aod
        self.latitude = latitude
        self.wavelengths = wavelengths
        assert time_units == 'days' or time_units == 'months' or time_units == 'years', 'Unrecognized time_units: ' \
                                                                                        'Use years, months, or days'

        if time_units == 'days':
            self.days = time
            self.years = self.days_to_years(time)
            self.months = self.years_to_months(self.years)

        elif time_units == 'months':
            self.months = time
            self.years = self.months_to_years(time)
            self.days = self.years_to_days(self.years)

        elif time_units == 'years':
            self.years = time
            self.days = self.years_to_days(time)
            self.months = self.years_to_months(time)

        if gm_aod is None:
            self.calc_gm_aod()
        else:
            self.gm_aod = gm_aod

    def save_as_nc(self, filename):

        dataset = nc.Dataset(filename, 'w', format='NETCDF4')

        time_dim = dataset.createDimension('time', None)
        lat_dim = dataset.createDimension('latitude', None)
        wave_dim = dataset.createDimension('wavelength', None)

        aod = dataset.createVariable('aod', 'f4', ('wavelength', 'time', 'latitude'))
        gm_aod = dataset.createVariable('gm_aod', 'f4', ('wavelength', 'time'))
        lat_var = dataset.createVariable('latitude', 'f4', 'latitude')
        days = dataset.createVariable('days', 'f4', ('time',))
        years = dataset.createVariable('years', 'f4', ('time',))
        wavelengths = dataset.createVariable('wavelength', 'f4', ('wavelength'))

        aod.units = 'unitless'
        gm_aod.units = 'unitless'
        days.units = 'days since 1850-1-1'
        years.units = 'years'
        lat_var.units = 'degrees north'
        wavelengths.units = 'nm'

        aod[:, :, :] = self.aod[:, :, :]
        gm_aod[:, :] = self.gm_aod[:, :]
        lat_var[:] = self.latitude[:]
        days[:] = self.days[:]
        years[:] = self.years[:]
        wavelengths[:] = self.wavelengths[:]

    def days_to_years(self, days):
        "Convert days since 1850-1-1 to Years"
        days_in_year = 365.2422
        return 1850 + days[:]/days_in_year

    def months_to_years(self, months):
        "Convert months since 1850-1-1 to Years"
        return [x/12 + 1850 for x in months]

    def years_to_months(self, years):
        "Convert years to months since 1850-1-1"
        return [(x-1850)*12 for x in years]

    def years_to_days(self, years):
        "Convert years into days since 1850-1-1"
        days_in_year = 365.2422
        return [days_in_year*(x - 1850) for x in years]

    def calc_gm_aod(self):
        """
        Calculates the weighted global mean AOD
        :return:
        """

        # Check if a particular wavelength is specified
        """
        if wavelength is None:  # Will average across bands if no wavelength given
            aod = np.zeros([len(self.data.lat), len(self.data.time)])
            # Calculate the weighted latitude data
            weighted_lat[:] = np.cos(self.data.lat[:] * np.pi / 180)
            weighted_lat[:] = weighted_lat[:] / sum(weighted_lat[:])
            # Average the aod across all bands
            aod = np.sum(self.data.aod_earth[:, :, :], 1) + np.sum(self.data.aod_sun[:, :, :], 1)
            aod = aod / (len(self.data.aod_earth[0, :, 0]) + len(self.data.aod_sun[0, :, 0]))
            # Fill the gm_aod with the properly weighed data
            gm_aod[:] = np.inner(np.sum(weighted_lat * aod[:, :]), weighted_lat)
        """
        # Create the gm aod array that will be filled
        self.gm_aod = np.zeros((len(self.wavelengths), len(self.years)))
        for index, wavelength in enumerate(self.wavelengths):  # if a wavelength is specified, only plot for that band
            # Create arrays to be filled
            weighted_lat = np.zeros(len(self.latitude))
            gm_aod = np.zeros(len(self.years))
            # Determine the band and aod data to use
            aod_in = self.aod
            # Calculate the weighted latitude data
            weighted_lat[:] = [np.cos(lat * np.pi / 180) for lat in self.latitude]
            weighted_lat[:] = weighted_lat[:] / sum(weighted_lat[:])
            # Fill the gm_aod with the properly weighed aod data
            gm_aod[:] = np.inner(aod_in[index, :, :], weighted_lat)
            self.gm_aod[index] = gm_aod
        self.gm_aod = np.ma.masked_where(self.gm_aod > 100, self.gm_aod)


def plot_gm_aod(dataset, time='years', fig=None, ax=None):
    # open the data
    nc = open_ncfile(dataset)
    wavelength = nc.variables['wavelength'][0]
    gm_aod = nc.variables['GM_AOD'][0, :]
    time_data = nc.variables[time][:]
    # Remove high value data masks
    gm_aod = np.ma.masked_where(gm_aod > 100, gm_aod)
    # Plot the data
    title = ("Global Mean AOD Trend at Wavelength: " + str(wavelength) + 'nm')
    xlabel = ("Time (" + time + ")")
    ylabel = ("Global Mean AOD at lambda = " + str(wavelength))
    fig = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel)
    fig.line(time_data, gm_aod)
    return fig


def save_aod_plot(dataset, divs=10, level_divs=None):
    """
    Builds and saves a single aod plot using matplotlib for the given band, time, and datatype.
    """
    aod = dataset['AOD'][:, :]
    time = dataset['Time'][:]
    lat = dataset['Latitude'][:]
    file = dataset['file']
    save_name = IMAGE_FOLDER + dataset['name'] + '.png'

    #  First check if this plot is already stored in images folder
    if os.path.isfile(save_name):
        return

    # aod has dims of time, latitude. This code (repurposed from plot_eva script)
    # uses time, wavelength, latitude
    z_data = np.zeros((len(time), 1, len(lat)))
    z_data[:, 0, :] = aod
    band = 0

    aod_scale = 1e2
    time_units = 'Years'
    lat_units = 'Degrees North'

    z_mean = aod_scale * z_data[:, band, :]

    # Set the data to be plotted
    x_data = time
    y_data = lat
    # Set the colorbar divisions - should start at 0 for aod
    if level_divs is None:
        level_divs = np.linspace(0, np.ceil(np.max(z_mean)), divs)

    # Make sure there is actually varying data here to be plotted, adjust division levels if not
    if max(level_divs) <= min(level_divs):
        level_divs = np.linspace(np.amin(z_mean), np.amax(z_mean) + 1e-14, 1)
    # Draw filled contours
    cmap = plt.cm.get_cmap('hot_r')
    cnplot = plt.contourf(x_data, y_data, z_mean.transpose(), divs,
                          cmap=cmap, extend='both')

    # Format x axis
    # -- Format x axis if time scale is too small for normal aod ticks (two years)
    max_x = np.ceil(max(time))  # Round up to nearest integer
    min_x = np.floor(min(time))  # Round down to nearest integer
    if max_x - min_x < 2:
        num_xticks = 5
        xtick_locations = np.arange(min_x, max_x, (max_x - min_x) / num_xticks)
        rounded_xticks = [round(elem, 2) for elem in xtick_locations]
        plt.xticks(xtick_locations, rounded_xticks, rotation=45)
    # Set the plotting labels
    colorbar_label = 'AOD x ' + "{:.0e}".format(aod_scale)
    xlabel = ('Time, ' + time_units)
    ylabel = ('Latitude, ' + lat_units)

    # Add colorbar
    cbar = plt.colorbar(cnplot, orientation='vertical')  # pad: distance between map and colorbar
    cbar.set_label(colorbar_label)  # add colorbar title string
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(dataset['name'])

    # Save the current figure to the current directory as a .png file.
    plt.savefig(save_name)
    plt.clf()


if __name__ == "__main__":

    folder = PROCESSED_DATA_FOLDER
    # Set data we will be looking at
    files = [folder + 'processed_tau_map.nc', folder + 'processed_ammann.nc', folder + 'processed_eVolv2k.nc',
             folder + 'processed_ICI5.nc', folder + 'processed_IVI2.nc', folder + 'processed_CMIP6.nc']
    names = ['GISS', 'Ammann', 'eVolv2k', 'ICI5', 'IVI2', 'CMIP6']
    colours = ['red', 'green', 'blue', 'black', 'orange', 'purple']
    authors = ["GISS", "NOAA", 'EVA Model', "NOAA", "Rutgers University", "Author Unknown"]
    links = ["https://data.giss.nasa.gov/modelforce/strataer/",
             "https://data.noaa.gov/dataset/dataset/noaa-wds-paleoclimatology-ammann-et-al-2003-monthly-volcanic-forcing-data-for-climat-1890-1999",
             "https://cera-www.dkrz.de/WDCC/ui/cerasearch/entry?acronym=eVolv2k_v3",
             "https://www.ncdc.noaa.gov/paleo-search/study/14168",
             "http://climate.envsci.rutgers.edu/IVI2/",
             "ftp://iacftp.ethz.ch/pub_read/luo/CMIP6/CMIP_1850_2014_extinction_550nm_strat_only_v3.nc"]


    # Process any missing files
    if not os.path.isfile(files[0]):
        tau_map_preprocessing()
    if not os.path.isfile(files[1]):
        ammann_preprocessing()
    if not os.path.isfile(files[2]):
        evolv2k_preprocessing()
    if not os.path.isfile(files[3]):
        ICI5_preprocessing()
    if not os.path.isfile(files[4]):
        IVI2_preprocessing()
    if not os.path.isfile(files[5]):
        CMIP6_preprocessing()

    # Sets the order of the datasets displayed (top buttons and legend)
    order = [2, 3, 4, 5, 1, 0] # -- Ordered by start date

    ####################################################################################################################
    # To add a new dataset, nothing beyond this point needs to be altered.
    ####################################################################################################################

    # Reorder the data as needed
    if order != [0, 1, 2, 3, 4, 5]:
        files = [files[i] for i in order]
        names = [names[i] for i in order]
        colours = [colours[i] for i in order]
        authors = [authors[i] for i in order]
        links = [links[i] for i in order]


    # Build a list of dictionaries containing the datasets and their relevant data
    Datasets = []
    for index, file in enumerate(files):
        nc = open_ncfile(files[index])
        current_set = {
            'name': names[index],
            'file': file,
            'nc data': nc,
            'AOD': nc.variables['aod'][0][:][:],
            'GM_AOD': nc.variables['gm_aod'][0][:],
            'Latitude': nc.variables['latitude'][:],
            'Time': nc.variables['years'][:]
        }
        Datasets.append(current_set)
    # Convert list to a dictionary for easy name lookup later
    Datasets = {names[i]: Datasets[i] for i in range(len(Datasets))}

    # Build a list of the plot tabs we are creating
    tabs = []

    # Set plot formatting
    title = "Global Mean SAOD Trend at Wavelength: 550nm"
    xlabel = "Time (Years)"
    ylabel = "Global Mean SAOD at lambda = 550nm"
    width = 1000
    height = 500

    # Set labels for each plot point
    TOOLTIPS = [("index", "$index"), ("(x,y)", "($x, $y)")]

    # Build multi-plot
    plot = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, tooltips=TOOLTIPS,  width=width,
                  tools=["xpan", "pan", "box_zoom", "reset"], height=height, x_range=Range1d(start=-50, end=2020))
    # Build lists to store the lines and their data sources for later
    lines = []
    sources = []
    for index, ds in enumerate(Datasets):
        # Pull the relevant data from the dictionary
        current_set = Datasets[names[index]]
        data = {'GM_AOD': current_set['GM_AOD'],
                'Time': current_set['Time']}
        # Structure the data into the style bokeh uses (pandas dataframes)
        df = pd.DataFrame(data)
        source = ColumnDataSource(data=df)

        # Create the plot, with both lines and points
        lines.append(plot.line('Time', 'GM_AOD', source=source, line_width=2, color=colours[index], legend_label=names[index]))

        # Store the source to be used later
        sources.append(source)

    # Add an interactive plot tool to select a date range
    # First, build the plot
    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    plot_height=200, plot_width=1000, y_range=plot.y_range, y_axis_type=None,
                    tools=["xpan", "pan", "box_zoom", "reset"], background_fill_color="#efefef")
    select.toolbar.logo = None

    # Build the range tool we will overlay on the plot
    range_tool = RangeTool(x_range=plot.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    # Add all the data lines to the plot, and store them (for setting visible later)
    lines2 = []
    for index, source in enumerate(sources):
        lines2.append(select.line('Time', 'GM_AOD', source=source, color=colours[index]))

    # Overlay the range tool on the plot
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool

    # Build a checkbox group to select the visible datasets
    # First, write the javascript callback that will set lines visible or not
    CB_callback = CustomJS(args=dict(lines=lines, lines2=lines2), code="""
    //console.log(cb_obj.active);
    for (var i in lines) {
        lines[i].visible = false;
        lines2[i].visible = false;
    }
    for (var i in cb_obj.active) {
        //console.log(cb_obj.active[i]);
        lines[cb_obj.active[i]].visible = true;
        lines2[cb_obj.active[i]].visible = true;
    }
""")
    # Build the checkbox group
    checkbox = CheckboxButtonGroup(active=list(range(len(lines))), labels=names)
    # Link it to the callback we built
    checkbox.js_on_click(CB_callback)

    # Add a text block for citing the data sources - Current Web Page cites in the html, so omitting this here.
    citation_text = ""
    """
    citation_text = "Data Sources: \n"
    for index, name in enumerate(names):
        citation_text = citation_text + '\'' + names[index] + "\': " + authors[index] + '. ' + links[index] + '\n'
    """
    citation_text = citation_text + "\nPlots created using the \'Bokeh\' plotting library.\nCreated by C.J. Ogilvie "\
                                    "under the supervision of Dr. Matthew Toohey at the University of Saskatchewan."
    citation = PreText(text=citation_text)

    # Combine the main plot, range select plot, and checkbox together on a single tab
    tabs.append(Panel(child=column(checkbox, plot, select), title="Global Mean SAOD"))

    ###################################################################################################################
    # Plot zonal AOD Data
    plots = []
    x_label = 'Time (Years)'
    y_label = 'Latitude (degrees north)'
    aod_tabs = []
    for index, ds in enumerate(Datasets):

        # Pull the relevant data from the dictionary
        current_set = Datasets[names[index]]
        aod = current_set['AOD'][:][:]
        lat = current_set['Latitude']
        time = current_set['Time']
        name = current_set['name']


        data = {'AOD': aod,
                'Latitude': lat,
                'Time': time}

        # Create the plot, with both lines and points
        n_divs = 10
        color = palettes.inferno(n_divs)
        color = color[::-1]  # reverse the color palette
        # Override the max value for files with outlier points
        if name == "GISS":
            high = 0.2
        elif name == "IVI2":
            high = 1.0
        else:
            high = round(0.75*np.max(aod), 1)

        mapper = LinearColorMapper(palette=color, low=0, high=high)
        levels = np.linspace(0, high, n_divs)

        if index < 3:
            x_range = [-500, 2020]
        else:
            x_range = [1850, 2020]

        p = figure(title=names[index], x_axis_label='Time (Years)', y_axis_label='Latitude (Deg. N)', x_range=x_range,
                   y_range=[-90, 90], width=1000, height=250, tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                   tools=["xpan", "xwheel_zoom", "reset"])
        p.yaxis.ticker = [-90, -60, -30, 0, 30, 60, 90]
        # For our ordering, divided into millennial (-500-2020) and historical (1850-2020):

        p.toolbar.active_scroll = p.select_one(WheelZoomTool)
        p.toolbar.logo = None

        # Using image will read lat in only first-to-last direction, if lat starts at +90 we need to flip the data
        if lat[0] > 0:
            aod = np.flip(aod, axis=1)
        # must give a vector of image data for image parameter
        image = [np.transpose(aod)]
        # Calculate width and height of image, assuming equal spacing for lat and time
        dh = 180
        dw = abs(time[-1]-time[0])
        p.image(image=image, x=np.min(time), dw=dw, y=np.min(lat), dh=dh, color_mapper=mapper)
        color_bar = ColorBar(color_mapper=mapper,
                             major_label_text_font_size="8pt",
                             ticker=FixedTicker(ticks=levels),
                             formatter=PrintfTickFormatter(format='%.2f'),
                             label_standoff=6,
                             location=(0, 0))

        p.add_layout(color_bar, 'right')

        plots.append(p)

    # Organize the plots into multiple tabs with a set number of plots each
    titles = ["Millennial", "Historical"]  # -- Our current order gives this organization
    num_plots = len(plots)
    page_index = 1
    while num_plots > 0:
        plots_per_page = 3
        first_index = (page_index - 1) * plots_per_page
        if num_plots >= plots_per_page:
            last_index = first_index + plots_per_page
        else:
            last_index = first_index + num_plots
        aod_tabs.append(Panel(child=column(plots[first_index:last_index]), title=titles[page_index-1]))
        page_index += 1
        num_plots -= plots_per_page

    tabs.append(Panel(child=Tabs(tabs=aod_tabs), title="Zonal Mean SAOD"))

    # Set the output html and display all tabs
    tabs = Tabs(tabs=tabs)
    layout = column(tabs, citation)
    output_file('AOD_Web_App.html', mode='inline')
    show(layout)