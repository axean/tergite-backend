# This code is part of Tergite
#
# (C) Chalmers Next Labs (2025)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Some Typing's to simplify reading the code"""
from typing import (
    Annotated,
    Any,
    Dict,
    Generic,
    Hashable,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import xarray as xr
from numpy import typing as npt
from xarray.core import dtypes

_T = TypeVar("_T", np.complexfloating, int)
_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V", bound="QDataArray")


class QDataArray(xr.DataArray, Generic[_T]):
    """A data array consisting of elements of type _T"""

    __slots__ = ()

    def __init__(
        self,
        data: npt.NDArray[_T] = dtypes.NA,
        coords: Union[
            Sequence[Union[Sequence[Any], pd.Index, "QDataArray"]],
            Mapping[Any, Any],
            None,
        ] = None,
        dims: Union[Hashable, Sequence[Hashable], None] = None,
        name: Optional[Hashable] = None,
        attrs: Optional[Mapping] = None,
        # internal parameters
        indexes: Optional[Mapping[Any, xr.Index]] = None,
        fastpath: bool = False,
    ):
        super().__init__(
            data=data,
            coords=coords,
            dims=dims,
            name=name,
            attrs=attrs,
            indexes=indexes,
            fastpath=fastpath,
        )

    @classmethod
    def from_xarray(cls, obj: xr.DataArray):
        """This converts xr.DataArray to QDataArray

        This is just to satisfy type checkers so that
        xarray.Dataset can be returned in functions that expect
        QDataset

        Args:
            obj: the xr.DataArray
        """
        return obj


class QDataset(xr.Dataset, Generic[_K, _V], Mapping[_K, _V]):
    """The dataset for the results returned on acquisition as mapping of _K: _V"""

    __slots__ = ()

    def __init__(
        self,
        data_vars: Optional[Mapping[_K, _V]] = None,
        coords: Optional[Mapping[Any, Any]] = None,
        attrs: Optional[Mapping[Any, Any]] = None,
    ):
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)

    @classmethod
    def from_xarray(cls, obj: xr.Dataset):
        """This converts xr.Dataset to QDataset

        This is just to satisfy type checkers so that
        xarray.Dataset can be returned in functions that expect
        QDataset

        Args:
            obj: the xr.Dataset
        """
        return obj


IQValue = np.complexfloating
RepetitionsByAcquisitionsMatrix = Annotated[QDataArray[IQValue], "repetition", "acq"]
"""xarray.DataArray of repetitions (rows) by acquisitions (columns)"""
# See: https://quantify-os.org/docs/quantify-scheduler/dev/user/user_guide.html#retrieve-acquisitions-through-instrumentcoordinator

QChannel = str
QExperimentName = str
QExperimentResult = QDataset[QChannel, RepetitionsByAcquisitionsMatrix]
QJobResult = Dict[QExperimentName, QDataset[QChannel, RepetitionsByAcquisitionsMatrix]]
