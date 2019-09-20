edgetpu.basic.edgetpu\_utils
============================

General utility APIs.

.. py:module:: edgetpu.swig.edgetpu_cpp_wrapper


Constants
---------

.. py:attribute:: edgetpu.basic.edgetpu_utils.EDGE_TPU_STATE_ASSIGNED

   Used with :meth:`ListEdgeTpuPaths` to list Edge TPU devices that are already associated with
   an instance of an inference engine (a :class:`~edgetpu.basic.basic_engine.BasicEngine`).


.. py:attribute:: edgetpu.basic.edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED

   Used with :meth:`ListEdgeTpuPaths` to list Edge TPU devices that are currently available.


.. py:attribute:: edgetpu.basic.edgetpu_utils.EDGE_TPU_STATE_NONE

   Used with :meth:`ListEdgeTpuPaths` to list all known Edge TPU devices.


.. py:attribute:: edgetpu.basic.edgetpu_utils.SUPPORTED_RUNTIME_VERSION

   The Edge TPU runtime version that's required by this Edge TPU library.


Functions
---------

.. autofunction:: edgetpu.basic.edgetpu_utils.ListEdgeTpuPaths

.. autofunction:: edgetpu.basic.edgetpu_utils.GetRuntimeVersion