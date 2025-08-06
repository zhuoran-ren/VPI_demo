/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            KU Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#ifndef CASADI_GHC_HPP
#define CASADI_GHC_HPP

#include "casadi/core/filesystem_impl.hpp"
#include <casadi/interfaces/ghc/casadi_filesystem_ghc_export.h>

/** \defgroup plugin_Filesystem_ghc Title
    \par

    \identifier{2d0} */

/** \pluginsection{Filesystem,ghc} */

/// \cond INTERNAL

namespace casadi {
  /** \brief \pluginbrief{Filesystem,ghc}

    Interface to ghc functionality

    @copydoc Filesystem_doc
    @copydoc plugin_Filesystem_ghc
    \author Joris Gillis
    \date 2025
  */
  class CASADI_FILESYSTEM_GHC_EXPORT Ghc : public Filesystem {
  public:
    /// A documentation string
    static const std::string meta_doc;
  };

} // namespace casadi

/// \endcond
#endif // CASADI_GHC_HPP
