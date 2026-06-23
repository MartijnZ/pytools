import os, time
import xml.etree.cElementTree as ET
import madernpytools.data_processing.DAQ as mdaq
import madernpytools.log as mlog
from madernpytools.backbone import ArgumentVerifier, EventPublisher, SimpleEventSubscriber, Timeout, IXML


class ModalTestConfiguration(IXML):

    def __init__(self,
                 test_description: str,
                 task_name: str = 'ModalTest',
                 chassis_name: str = 'MadernRD',
                 hammer_sensor: mdaq.PCBImpactHammerSensor = None,
                 acceleration_sensors: mdaq.SensorList = None,
                 sampling_rate: int = 25600,
                 impact_trigger=20.0, log_size=51200,
                 impact_timeout=30,
                 test_timeout=30
                 ):
        """

        :param test_description: Description of modal test
        :param task_name: Name of the data acquisition task (default 'ModalTest')
        :param chassis_name: Name of the NI-DAQ chassis
        :param hammer_sensor: Impact hammer sensor object
        :type hammer_sensor: mdaq.Sensor
        :param acceleration_sensors:  Acceleration sensor object(s).
        :type acceleration_sensors: list
        :param sampling_rate: Sampling rate for modal test (default: 25600Hz)
        :param impact_trigger: Impact value for which logging is activated (default: 20 [N])
        :param log_size: Number of samples to log (default 51200)
        :param impact_timeout: Time before impact timeout is raised (default 30 sec)
        """

        # Default values:
        default_hammer = mdaq.PCBImpactHammerSensor(device_name='MadernRDMod1', channel_index=1, name='impact hammer')
        default_acc = mdaq.SensorList([mdaq.PCBAccelerationSensor(device_name='MadernRDMod1', channel_index=0,
                                                                  name='acc1'),
                                       mdaq.PCBAccelerationSensor(device_name='MadernRDMod1', channel_index=1,
                                                                  name='acc2')
                                       ])

        # Task configuration
        self.task_name = ArgumentVerifier(str, task_name).verify(task_name)
        self.test_description = ArgumentVerifier(str, test_description).verify(test_description)
        self.chassis_name = ArgumentVerifier(str, chassis_name).verify(chassis_name)
        self.hammer = ArgumentVerifier(mdaq.PCBImpactHammerSensor, default_hammer).verify(hammer_sensor)

        # Sensors & Timing
        self.acceleration_sensors = ArgumentVerifier(list, default_acc).verify(acceleration_sensors)
        self.sampling_rate = ArgumentVerifier(int, sampling_rate).verify(sampling_rate)

        # Log configuration:
        self.impact_trigger = ArgumentVerifier((float, int), impact_trigger).verify(impact_trigger)
        self.impact_timeout = ArgumentVerifier((int, float), impact_timeout).verify(impact_timeout)
        self.test_timeout = ArgumentVerifier((int, float), test_timeout).verify(test_timeout)

        self.log_size = ArgumentVerifier(int, log_size).verify(log_size)
        self.test_description = ArgumentVerifier(str, test_description).verify(test_description)

    @property
    def duration(self):
        return float(self.log_size)/float(self.sampling_rate)

    @staticmethod
    def from_xml(root: ET.ElementTree):
        """ Extracts ModalTestConfiguration from xml file
        :param xml_tree: xml tree containing the experiment configuration elements
        :return: Experimental configuraiton
        """

        xml_tree = root.find('Modal_Settings')

        # Get sensors:
        acc_sens = mdaq.SensorList()
        hammer = None

        for child in xml_tree.find('sensor_settings'):
            # Identify type:
            print('Child: ', child)
            print('Child type: ', child.get('Type'))
            child_type = mdaq.DAQClassFactory.get(child.get('Type'))

            # Extract dict:
            if issubclass(child_type, IXML):
                if child_type is mdaq.PCBImpactHammerSensor:
                    hammer = child_type.from_xml(child, mdaq.DAQClassFactory)
                elif child_type is mdaq.PCBAccelerationSensor:
                    acc_sens.append(child_type.from_xml(child, mdaq.DAQClassFactory))

        #for sensor_setting in xml_tree.find('sensor_settings').iter('sensor'):
        #    if sensor_setting.find('type').text == 'acceleration':
        #        acc_sens.append(mdaq.PCBAccelerationSensor.from_xml(sensor_setting))
        #    elif sensor_setting.find('type').text == 'impact_hammer':
        #        hammer = mdaq.PCBAccelerationSensor.from_xml(sensor_setting)

        acc_sens = None if (len(acc_sens) == 0) else acc_sens

        return ModalTestConfiguration(
                    test_description=xml_tree.find('log_settings').find('test_description').text,
                    task_name=xml_tree.find('DAQ_Settings').find('task_name').text,
                    chassis_name=xml_tree.find('DAQ_Settings').find('chassis_name').text,
                    sampling_rate=int(xml_tree.find('DAQ_Settings').find('sampling_rate').text),
                    impact_trigger=float(xml_tree.find('log_settings').find('impact_trigger').text),
                    impact_timeout=int(xml_tree.find('log_settings').find('impact_timeout').text),
                    test_timeout=int(xml_tree.find('log_settings').find('test_timeout').text),
                    log_size=int(xml_tree.find('log_settings').find('log_size').text),
                    acceleration_sensors=acc_sens,
                    hammer_sensor=hammer
                  )

    def to_xml(self):
        """
        :return:
        """
        settings = ET.Element('Modal_Settings')

        # DAQ settings:
        daq_settings = ET.SubElement(settings, 'DAQ_Settings')
        for attr in ['task_name', 'chassis_name', 'sampling_rate']:
            xml_item = ET.SubElement(daq_settings, attr)
            xml_item.text = str(getattr(self, attr))

        # Sensor settings:
        sensor_settings = ET.SubElement(settings, 'sensor_settings')
        for s in self.acceleration_sensors:
            sensor_settings.append(s.to_xml())
        sensor_settings.append(self.hammer.to_xml())

        # Log Settings
        log_settings = ET.SubElement(settings, 'log_settings')
        for attr in ['test_description', 'impact_trigger', 'impact_timeout', 'test_timeout', 'log_size']:
            xml_item = ET.SubElement(log_settings, attr)
            xml_item.text = str(getattr(self, attr))

        return settings


class AbstractModalTest(object):

    def wait_until_done(self):
        raise NotImplementedError()

    def do_test(self, filename, verbose=True):
        """Perform a modal test

        :param filename: Filename of the data log
        :param verbose: If true progress messages are printed.
        :return: None
        """
        raise NotImplementedError()

    def abort(self, reason=''):
        """ Abort the current task

        :param reason: Reason for aborting (this is printed to commandline)
        :return:
        """
        raise NotImplementedError()

    def do_grid_test(self, base_filename, n_locations, n_reps, i_measurement, verbose=True):
        """

        :param base_filename: The filename to use for generated logs, if name is log.csv, logs with names log_I1S2_R1.csv
        are generated
        :param n_locations: Number of locations in the grid
        :param n_reps: Number of repetitions for each grid point
        :param i_measurement: The grid location on which the vibration pickup is placed
        :param verbose: If true, progress messages are printed.
        :return:
        """
        raise NotImplementedError()

    @property
    def log(self):
        raise NotImplementedError()

    @property
    def event_impact_timeout(self):
        """

        :return: Impact timeout event
        """
        raise NotImplementedError()

    @property
    def event_test_timeout(self):
        """

        :return: Test timeout event
        """
        raise NotImplementedError()

    @property
    def event_impact_request(self):
        raise NotImplementedError()

    @property
    def event_test_finished(self):
        raise NotImplementedError()

    @property
    def event_impact_detected(self):
        """

        :return: Impact detected event
        """
        raise NotImplementedError

    @property
    def task_configuration(self):
        raise NotImplementedError()

    @task_configuration.setter
    def task_configuration(self, value):
        raise NotImplementedError()


    @property
    def event_file_saved(self):
        """

        :return: File saved event
        """
        raise NotImplementedError


class ModalTest(AbstractModalTest):

    def __init__(self, task_configuration: ModalTestConfiguration):
        """ A class for handling modal testing.

        :param task_configuration: modal test configuration
        """

        # Variables:
        self._task_configuration = ArgumentVerifier(ModalTestConfiguration, task_configuration
                                                    ).verify(task_configuration)
        self._task = mdaq.AbstractMadernTask()
        self._log = mlog.AbstractLog()
        self._task_running = False
        self._trigger_index = None
        self._impact_timeout = Timeout()
        self._test_timeout = Timeout()

        # Setup objects (do not change order: log initialization depends on task initialization!)
        self._setup_task(self.task_configuration)
        self._setup_log(self.task_configuration)

        # Event publishers:
        self._event_impact_request = EventPublisher()
        self._event_test_finished = EventPublisher(data_type=mlog.AbstractLog)

        # Connect events:
        self._task.event_new_data.connect(self._log)        
        self.event_impact_detected.connect(self._impact_timeout)
        self.event_test_finished.connect(self._test_timeout)

    def wait_until_done(self):
        """ Wait until current task is done, timeout is raised after expected task duration is exceeded by 10 seconds
        :returns None
        """

        if self._task_running:
            # Expected duration
            duration = float(self.task_configuration.log_size)/float(self.task_configuration.sampling_rate)

            # Wait for timeout:
            self._test_timeout.wait_for_timeout(duration + self.task_configuration.test_timeout)

    def _setup_log(self, task_config: ModalTestConfiguration):
        """ private function to setup logging

        :param task_config:
        :return:
        """
        info = mlog.LogInfo(description=task_config.test_description,
                            sampling_rate=self._task.sampling_rate,
                            signal_header=self._task.signals.signal_names
                            )
        self._log = mlog.Log(info)

        # Log trigger:
        self._log.trigger.set_trigger(signal_index=self._trigger_index, value=task_config.impact_trigger)
        self._log.limiter.set(task_config.log_size)

    def _setup_task(self, task_config: ModalTestConfiguration):
        """ Set-up task according to task_config

        :param task_config:  Task configuration object
        :return: None
        """

        if isinstance(self._task, mdaq.ContinuousMadernTask):
            self._task.close()

        # Create task
        self._task = mdaq.ContinuousMadernTask(task_name=task_config.task_name,
                                               chassis_name=task_config.chassis_name)

        # Add sensors:
        for acc_sens in task_config.acceleration_sensors:
            self._task.add_sensor(acc_sens)
        self._task.add_sensor(task_config.hammer)

        self._trigger_index = task_config.hammer.signal.data_index
        self._task.configure(sampling_rate=task_config.sampling_rate, buffer_size=int(100e3))

        # After staring a NI-DAQ, the first batch of samples doesn't seem to be ok. To prevent bad measurements
        # we run the task once:
        self._task.start()
        time.sleep(0.5)
        self._task.stop()

    def _cb_test_finished(self, filename, verbose=False):
        """ Callback after test has finished, saves the log to filename

        :param filename: Filename to store the log into
        :param verbose: If true, function prints progress
        :return: None
        """
        if verbose:
            print("Logging finished, saving log to {0}...".format(filename))
        self._log.save(filename)
        if verbose:
            print("log saved")
        self._task.stop()
        self._task_running = False
        self.event_test_finished.raise_event(self._log)

    def do_test(self, filename, verbose=True):
        """Perform a modal test

        :param filename: Filename of the data log
        :param verbose: If true progress messages are printed.
        :return: None
        """

        if not self._task_running:
            self._task_running = True
            sub_timeout = SimpleEventSubscriber(lambda _, reason: self.abort(reason), reason='Trigger timed out')
            sub_triggered = SimpleEventSubscriber(lambda _: print('Impact detected'))

            # Prepare for experiment:
            self._task.reset()
            self._log.reset()
            self._log.limiter.set_limit_callback(self._cb_test_finished,             # Callback
                                                 filename=filename,
                                                 verbose=verbose  # keyword arguments
                                                 )

            # Do experiment:
            self._task.start()

            # Request impact:
            self.event_impact_request.raise_event()

            if verbose:
                self._log.trigger.event_timeout.connect(sub_timeout)
                self._log.trigger.event_triggered.connect(sub_triggered)

            # Wait for impact:
            self._log.trigger.wait_for_trigger(timeout=self.task_configuration.impact_timeout)
            self._test_timeout.start(timeout=self.task_configuration.duration + self.task_configuration.test_timeout)

            if verbose:
                # Disconnect
                self._log.trigger.event_timeout.disconnect(sub_timeout)
                self._log.trigger.event_triggered.disconnect(sub_triggered)               
        else:
            raise RuntimeWarning("Test is already running")

    def abort(self, reason=''):
        """ Abort the current task

        :param reason: Reason for aborting (this is printed to commandline)
        :return:
        """
        self._task.stop()
        self._task_running = False

        if reason != '':
            print('Task aborted')

    def do_grid_test(self, base_filename, n_locations, n_reps, i_measurement, verbose=True):
        """

        :param base_filename: The filename to use for generated logs, if name is log.csv, logs with names log_I1S2_R1.csv
        are generated
        :param n_locations: Number of locations in the grid
        :param n_reps: Number of repetitions for each grid point
        :param i_measurement: The grid location on which the vibration pickup is placed
        :param verbose: If true, progress messages are printed.
        :return:
        """

        path, ext = os.path.splitext(base_filename)
        for n in range(1, n_locations+1):
            if verbose:
                print('----- Location {0}------'.format(n))

            for k in range(n_reps):
                fn = '{0}_I{1}S{2}_R{3}{4}'.format(path, n, i_measurement, k, ext)

                if verbose:
                    print('* experiment {0}'.format(fn))

                self.do_test(fn, verbose)
                self.wait_until_done()

        if verbose:
            print('Grid experiment finished.')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.abort()                        
        self._task.close()

    @property
    def log(self):
        return self._log

    @property
    def event_impact_timeout(self):
        """

        :return: Impact timeout event
        """
        return self._impact_timeout.event_timeout

    @property
    def event_test_timeout(self):
        """

        :return: Test timeout event
        """
        return self._test_timeout.event_timeout
        
    @property
    def event_impact_detected(self):
        """

        :return: Impact detected event
        """
        return self._log.trigger.event_triggered

    @property
    def event_impact_request(self):
        return self._event_impact_request

    @property
    def event_test_finished(self):
        return self._event_test_finished

    @property
    def event_file_saved(self):
        """

        :return: File saved event
        """
        return self._log.event_file_saved

    @property
    def task_configuration(self):
        return self._task_configuration

    @task_configuration.setter
    def task_configuration(self, value):
        self._task_configuration = value


class SimulateModalTest(object):

    def __init__(self, task_configuration: ModalTestConfiguration):
        """ A class for handling modal testing.

        :param task_configuration: modal test configuration
        """

        # Variables:
        self.task_configuration = ArgumentVerifier(ModalTestConfiguration, task_configuration
                                                   ).verify(task_configuration)
        self._log = mlog.AbstractLog()
        self._task_running = False
        self._trigger_index = None
        self._impact_timeout = Timeout()
        self._test_timeout = Timeout()

        # Setup objects (do not change order: log initialization depends on task initialization!)
        self._setup_log(self.task_configuration)

        # Event publishers:
        self._event_impact_request = EventPublisher()
        self._event_test_finished = EventPublisher(data_type=mlog.AbstractLog)

        # Connect events:
        self.event_impact_detected.connect(self._impact_timeout)
        self.event_test_finished.connect(self._test_timeout)

    def wait_until_done(self):
        """Simulate wait"""

        if self._task_running:
            # Expected duration
            duration = float(self.task_configuration.log_size)/float(self.task_configuration.sampling_rate)

            # Wait for timeout:
            self._test_timeout.wait_for_timeout(duration + self.task_configuration.test_timeout)

    def _setup_log(self, task_config: ModalTestConfiguration):
        """ private function to setup logging

        :param task_config:
        :return:
        """
        self._log = mlog.Log(log_info=mlog.LogInfo('', signal_header=['', ''], sampling_rate=1))
        self._log.trigger.set_trigger(signal_index=0, value=10.0)
        self._log.limiter.set(task_config.log_size)

    def do_test(self, filename, verbose=True):
        """Perform a modal test

        :param filename: Filename of the data log to load during simulation
        :param verbose: If true progress messages are printed.
        :return: None
        """

        if not self._task_running:
            self._task_running = True
            sub_timeout = SimpleEventSubscriber(lambda _, reason: self.abort(reason), reason='Trigger timed out')
            sub_triggered = SimpleEventSubscriber(lambda _: print('Impact detected'))

            time.sleep(1.5)
            self.event_impact_request.raise_event()

            if verbose:
                print("Impact tool to start measurement...")
                self._log.trigger.event_timeout.connect(sub_timeout)
                self._log.trigger.event_triggered.connect(sub_triggered)

            # Raise triggered event
            self._test_timeout.start(timeout=self.task_configuration.duration + self.task_configuration.test_timeout)
            time.sleep(1.5)
            self._log.event_triggered.raise_event()

            if verbose:
                # Disconnect
                self._log.trigger.event_timeout.disconnect(sub_timeout)
                self._log.trigger.event_triggered.disconnect(sub_triggered)

            filename = './data/ema_simulation_data.csv'
            info, data = mlog.CSVLogReader().read(filename=filename)
            self._log = mlog.Log(info)
            self._log.data = data
            self.event_test_finished.raise_event(self._log)
            self._task_running = False
        else:
            raise RuntimeWarning("Test is already running")

    def abort(self, reason=''):
        """ Abort the current task

        :param reason: Reason for aborting (this is printed to commandline)
        :return:
        """
        self._task_running = False

        if reason != '':
            print('Task aborted')

    @property
    def log(self):
        return self._log

    @property
    def event_impact_timeout(self):
        """

        :return: Impact timeout event
        """
        return self._impact_timeout.event_timeout

    @property
    def event_test_timeout(self):
        """

        :return: Test timeout event
        """
        return self._test_timeout.event_timeout

    @property
    def event_impact_detected(self):
        """

        :return: Impact detected event
        """
        return self._log.trigger.event_triggered

    @property
    def event_impact_request(self):
        return self._event_impact_request

    @property
    def event_test_finished(self):
        return self._event_test_finished

    @property
    def event_file_saved(self):
        """

        :return: File saved event
        """
        return self._log.event_file_saved


if __name__ == "__main__":

    hammer = mdaq.PCBImpactHammerSensor(device_name='MadernRD-WifiMod1', channel_index=1, name='impact hammer')
    default_acc = mdaq.SensorList([mdaq.PCBAccelerationSensor(device_name='MadernRD-WifiMod1', channel_index=0,
                                                              name='acc1'),
                                   mdaq.PCBAccelerationSensor(device_name='MadernRD-WifiMod1', channel_index=1,
                                                              name='acc2')
                                   ])

    test_config = ModalTestConfiguration(test_description='Test',
                                         task_name='ModalTest',
                                         chassis_name='MadernRD-Wifi',
                                         hammer_sensor=hammer,
                                         acceleration_sensors=default_acc,
                                         sampling_rate=25600,
                                         impact_trigger=20.0,
                                         log_size=51200,
                                         impact_timeout=30,
                                         test_timeout=30)



    #filename='ema_config.xml'
    #if filename != '':
    #    ET.ElementTree(test_config.to_xml()).write(filename)

    test_config.impact_trigger = 20


    # List settings:
    for key, item in test_config.__dict__.items():
        print('{0}: {1}'.format(key, item))

    # List task settings:
    with ModalTest(test_config) as my_test:
        print('Verbose test')
        val = input('Enter filename and press [Enter] to start test sequence: ')
        my_test.do_test(filename='{0}.csv'.format(val))
        my_test.wait_until_done()
        
        print('Non verbose test')
        sub_impact_request = SimpleEventSubscriber(lambda _: print("Impact request event raised"))
        sub_impact_detected = SimpleEventSubscriber(lambda _: print("Trigger event raised"))
        sub_finished = SimpleEventSubscriber(lambda _: print("Finished event raised"))
        my_test.event_impact_request.connect(sub_impact_request)
        my_test.event_impact_detected.connect(sub_impact_detected)
        my_test.event_test_finished.connect(sub_finished)
        
        val = input('Enter filename and press [Enter] to start test sequence: ')
        my_test.do_test(filename='{0}.csv'.format(val), verbose=False)
        my_test.wait_until_done()
        
