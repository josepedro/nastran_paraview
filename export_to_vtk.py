# coding: utf-8
from numpy import zeros, searchsorted, arange, unique, array, insert, delete, where, array_split, mgrid, cumsum
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import sys

#def pack_nodes(fmt, data):
    #return ''


ETYPE_MAP = {
    # line
    'CDAMP1' : 3,
    'CDAMP2' : 3,
    'CDAMP3' : 3,
    'CDAMP4' : 3,
    'CELAS1' : 3,
    'CELAS2' : 3,
    'CELAS3' : 3,
    'CELAS4' : 3,
    'CBAR' : 3,
    'CBEAM' : 3,
    'CROD' : 3,
    'CONROD' : 3,
    'CTUBE' : 3,

    'CTRIA3' : 5, # triangle
    'CQUAD4' : 9,  # quad
    'CSHEAR' : 9,  # quad

    # quadratic
    'CTRIA6' : 22,  # quadratic triangle
    #'CQUAD8' : 23/28/30,

    'CTETRA' : 10,
    'CPENTA' : 13, # wedge
    'CPYRAM' : 14,
    'CHEXA' : 12, # hex

    # quadratic solids
    #'CTETRA' : 64,
    #'CPENTA' : 65, # wedge
    #'CPYRAM' : 66,
    #'CHEXA' : 67, # hex
}



def pack_int_array(fmt, data):
    return ' '.join([str(val) for val in data]) + '\n'

def pack_float_1d_array(fmt, data):
    return ' '.join([str(val) for val in data.ravel()]) + '\n'

def pack_float_3d_array(fmt, data):
    msg = ''
    for datai in data[0, :, :]:
        msgi = ''
        for dataii in datai:
            msgi += '%s ' % dataii
        msg += msgi[:-1] + '\n'
    return msg #+ '\n\n'

def pack_float_2d_array(fmt, data):
    msg = ''
    for datai in data:
        msgi = ''
        for dataii in datai:
            msgi += '%s ' % dataii
        msg += msgi[:-1] + '\n'
    return msg #+ '\n'

###----Modified by Marco. The original line is commented. Section start

def estraiel1(risultati):
    eiddi =  risultati.element
    return eiddi
def estraiel2(risultati):
    eiddi1 = risultati.element_node
    eiddi2 = eiddi1[:,0]
    eiddi = unique(eiddi2)
    return eiddi
def estraiel3(risultati):
    eiddi1 = risultati.element_layer[:, 0]
    eiddi = unique(eiddi1)
    return eiddi



def inseriscivaloriele3d(arraydati, dato, modello, chiave):
    elementi = array(list(modello.elements.keys()))
    ezzeri = zeros(len(elementi), float)
    eleres = estraiel2(arraydati[chiave])

    posizioneele = searchsorted(elementi, eleres)


    aggiunta = insert(ezzeri, posizioneele, dato)
    
    postidovetogliere = posizioneele + arange(len(posizioneele)) + 1
    uscita = delete(aggiunta, postidovetogliere)
        

    return uscita



def quasinumpyreplace(partenza, dove, cosa):
    aggiunta = insert(partenza, dove, cosa)
    postidovetogliere = dove + arange(len(dove)) + 1
    uscita = delete(aggiunta, postidovetogliere)
    return uscita  
###----Modified by Marco. Section end





#def pack(fmt, data):
#    return ''

def export_to_vtk(model_name: str) -> None:
    bdf_filename = model_name + '.bdf'
    op2_filename = model_name + '.op2'
    vtk_filename = model_name + '.vtk'
    bdf_model, op2_model = export_to_vtk_filename(bdf_filename, op2_filename, vtk_filename)
    bdf_model.log.info('finished exporting %s' % vtk_filename)
    return bdf_model, op2_model

def export_to_vtk_filename(bdf_filename: str, op2_filename: str, vtk_filename: str,
                           debug: bool=False, log=None) -> None:
    """exports a BDF/OP2 to VTK"""
    bdf_model = BDF(debug=debug, log=log)
    bdf_model.read_bdf(bdf_filename)
    op2_model = OP2(debug=debug, log=log)
    op2_model.read_op2(op2_filename)

    out = bdf_model.get_card_ids_by_card_types()


###----Modified by Marco. The original line is commented. Section start
            
    
    
    grids = sorted(out['GRID'])
    spoint = sorted(out['SPOINT'])
    epoint = sorted(out['EPOINT'])
    ngrid = len(grids)
    nspoint = len(spoint)
    nepoint = len(epoint)
    nnodes = ngrid + nspoint + nepoint

    ncrod = len(out['CROD'])
    nconrod = len(out['CONROD'])
    nctube = len(out['CTUBE'])
    ncbeam = len(out['CBEAM'])
    ncbar = len(out['CBAR'])
    nline = ncrod + nconrod + nctube + ncbeam + ncbar

    ncelas1 = len(out['CELAS1'])
    ncelas2 = len(out['CELAS2'])
    ncelas3 = len(out['CELAS3'])
    ncelas4 = len(out['CELAS4'])

    ncdamp1 = len(out['CDAMP1'])
    ncdamp2 = len(out['CDAMP2'])
    ncdamp3 = len(out['CDAMP3'])
    ncdamp4 = len(out['CDAMP4'])
    n0d = (ncelas1 + ncelas2 + ncelas3 + ncelas4 +
           ncdamp1 + ncdamp2 + ncdamp3 + ncdamp4)

    nctria3 = len(out['CTRIA3'])
    ncquad4 = len(out['CQUAD4'])
    nctria6 = len(out['CTRIA6'])
    ncquad8 = len(out['CQUAD8'])
    ncshear = len(out['CSHEAR'])
    nshell = nctria3 + ncquad4 + nctria6 + ncquad8 + ncshear

    nctetra4 = len(out['CTETRA'])
    ncpyram5 = len(out['CPYRAM'])
    ncpenta6 = len(out['CPENTA'])
    nchexa8 = len(out['CHEXA'])
    nctetra10 = 0
    ncpyram8 = 0
    ncpenta15 = 0
    nchexa20 = 0
    nsolid = (nctetra4 + ncpyram5 + ncpenta6 + nchexa8 +
              nctetra10 + ncpyram8 + ncpenta15 + nchexa20)

    #nelements = n0d + nline + nshell + nsolid
    nelements = 0
    etypes = [
        'CELAS1', 'CELAS2', 'CELAS3', 'CELAS4',
        'CDAMP1', 'CDAMP2', 'CDAMP3', 'CDAMP4',
        'CROD', 'CONROD', 'CTUBE',
        'CBAR', 'CBEAM',
        'CFAST', 'CBUSH', 'CBUSH1D', 'CBUSH2D',

        'CTRIA3', 'CQUAD4', 'CTRIA6', 'CQUAD8', 'CSHEAR',

        'CTETRA', 'CPENTA', 'CPYRAM', 'CHEXA',
    ]
    assert len(etypes) == len(set(etypes)), 'there are duplicate etypes'
    for etype in etypes:
        if etype in out:
            nelementsi = len(out[etype])
            nelements += nelementsi
    nproperties = nelements

    bdf_nelements = bdf_model.nelements

    # SPOINT & EPOINT are implicitly defined
    xyz_cid0 = zeros((nnodes, 3), dtype='float32')
    
###----Modified by Marco. The original line is commented. Section start
#    nids = zeros(nnodes, dtype='float32')
    nids = zeros(nnodes, dtype='int32')
###----Modified by Marco. Section end

###----Modified by Marco. The original line is commented. Section start
#    nids = zeros(nnodes, dtype='float32')
##Element array
    elementi = list(bdf_model.elements.keys())
    eidszero = zeros(bdf_nelements, dtype='float32')

###----Modified by Marco. Section end


    i = 0
    for i, nid in enumerate(grids):
        xyz_cid0[i, :] = bdf_model.nodes[nid].get_position()
    nids[:ngrid] = grids
    if nspoint:
        nids[i:i+nspoint] = spoint
    if nepoint:
        nids[i+nspoint:] = epoint

    nelements = n0d + nline + nshell + nsolid
    nmaterials = nelements

    eids = zeros(nelements, dtype='int32')

    cell_types = zeros(nelements, dtype='int32')
    pids = zeros(nelements, dtype='int32')
    mids = zeros(nelements, dtype='int32')


    with open(vtk_filename, 'w') as vtk_file:
        vtk_file.write('# vtk DataFile Version 3.1\n')
        vtk_file.write('created by pyNastran\n')
        #vtk_file.write('BINARY\n')
        vtk_file.write('ASCII\n')
        vtk_file.write('DATASET UNSTRUCTURED_GRID\n')

        nid_fmt = '%ii' % nnodes
        xyz_fmt = '%ii' % (nnodes * 3)
        vtk_file.write('POINTS %i float\n' % nnodes)
        vtk_file.write(pack_float_2d_array(xyz_fmt, xyz_cid0))

        eid_fmt = '%ii' % nelements

        # we'll add 1 to the slot count of each
        # so for a single CROD, it has 2 nodes and 1 extra value (to indicate it's a line)
        # for a total of 3
        nline_slots = nline * 3
        nshell_slots = 4 * nctria3 + 5 * (ncquad4 + ncshear) + 7 * nctria6 + 9 * ncquad8
        nsolid_slots = 5 * nctetra4 + 6 * ncpyram5 + 7 * ncpenta6 + 9 * nchexa8
        bdf_model.log.debug('nline=%s nshell=%s nsolid=%s' % (nline, nshell, nsolid))
        assert nelements == bdf_nelements, 'nelements=%s bdf_model.nelements=%s card_count=\n%s' % (
            nelements, bdf_nelements, bdf_model.card_count)

###----Modified by Marco. The original line is commented. Section start
#---There are errors in total node count: nelements_slots is wrong (for example you consider CELAS or CBUSH
#  as as one-node elements while they may have two nodes). It is not a problem if you import the .vtk file
# in Gsmh, while Paraview crashes.

        nelements_slots = nline_slots + nshell_slots + nsolid_slots


#--This loop calculates the total number of nodes related to each element. It is much slower than your approch, so sure you can be better.
        elementi = list(bdf_model.elements.keys())
        nnn = 0
        
        for jj in elementi:
            topo = bdf_model.Element(jj).type
            nodis = len(bdf_model.Element(jj).nodes)
            tipo = topo + str(nodis)
            num_nodi_el = nodis
            nodi_el = bdf_model.Element(jj).nodes 
            nnn  = nnn + nodis
        chiusa = nnn + len(elementi)
        


        vtk_file.write('CELLS %i %i\n' % (nelements, chiusa))

###----Modified by Marco. Section end

        i = 0
        for eid, elem in sorted(bdf_model.elements.items()):
            etype = ETYPE_MAP[elem.type]
            nids2 = searchsorted(nids, elem.node_ids)

            nnodesi = len(nids2)
            vtk_file.write('%i %s\n' % (nnodesi, str(nids2)[1:-1]))
            if elem.type in ['CTETRA', 'CPENTA', 'CHEXA', 'CPYRAM', 'CBEAM', 'CROD', 'CBAR']:
                pid = elem.Pid()
                mid = elem.Mid()
            elif elem.type in ['CELAS1', 'CELAS2', 'CELAS3', 'CELAS4',
                               'CDAMP1', 'CDAMP2', 'CDAMP3', 'CDAMP4', 'CBUSH', 'CFAST']:
                pid = elem.Pid()
                mid = 0
            elif elem.type in ['CQUAD4', 'CQUAD8', 'CQUADX', 'CQUADX8', 'CQUAD',
                               'CTRIA3', 'CTRIA6', 'CTRIAX', 'CTRIAX6', 'CSHEAR']:
                pid = elem.Pid()
                prop = elem.pid_ref
                if prop.type in ['PCOMP', 'PCOMPG']:
                    mid = prop.Mid(0)
                elif prop.type in ['PSHELL']:
                    mid = prop.Mid1()
                elif prop.type in ['PSHEAR']:
                    mid = prop.Mid()
                else:
                    raise NotImplementedError(prop)
            elif elem.type in ['CONROD']:
                pid = 0
                mid = elem.Mid()
            else:
                raise NotImplementedError(elem)

            eids[i] = eid
            pids[i] = pid
            mids[i] = mid
            cell_types[i] = etype
            i += 1
        assert nelements == bdf_nelements, 'i=%s nelements=%s bdf.nelements=%s' % (i, nelements, bdf_nelements)

        #vtk_file.write('\n')
        vtk_file.write('CELL_TYPES %i\n' % nelements)
        vtk_file.write(pack_int_array(eid_fmt, cell_types))
        vtk_file.write('\n')

        vtk_file.write('POINT_DATA %i\n' % nnodes)
###----Modified by Marco. The original line is commented. Section start
#        vtk_file.write('NodeID %i float\n' % nnodes)
        vtk_file.write('\n')
        vtk_file.write('\n')
        vtk_file.write('SCALARS NodeID int\n')
        vtk_file.write('LOOKUP_TABLE NodeID\n')
###----Modified by Marco. Section end
        vtk_file.write(pack_int_array(nid_fmt, nids))

        fmt = '%si' % nelements
        if nelements:
        
###----Modified by Marco. The original line is commented. Section start
#            vtk_file.write('ElementID %i float\n' % nelements)
            vtk_file.write('\n')
            vtk_file.write('CELL_DATA %i\n' % nelements)
            vtk_file.write('\n')
            vtk_file.write('\n')
            vtk_file.write('SCALARS ElementID int\n')
            vtk_file.write('LOOKUP_TABLE ElementID\n')
###----Modified by Marco. Section end
            vtk_file.write(pack_int_array(eid_fmt, eids))
        if nproperties:
###----Modified by Marco. The original line is commented. Section start
#            vtk_file.write('PropertyID %i float\n' % nproperties)
            vtk_file.write('\n')
            vtk_file.write('SCALARS PropertyID int\n')
            vtk_file.write('LOOKUP_TABLE PropertyID\n')

###----Modified by Marco. Section end            
            vtk_file.write(pack_int_array(eid_fmt, pids))
        if nmaterials:
###----Modified by Marco. The original line is commented. Section start
#            vtk_file.write('MaterialID %i float\n' % nmaterials)
            vtk_file.write('\n')
            vtk_file.write('SCALARS MaterialID int\n')
            vtk_file.write('LOOKUP_TABLE MaterialID\n')

###----Modified by Marco. Section end
            vtk_file.write(pack_int_array(eid_fmt, mids))
            vtk_file.write('\n\n')


        nodal_cases = [op2_model.eigenvectors, op2_model.displacements,
                       op2_model.velocities, op2_model.accelerations, op2_model.spc_forces,
                       op2_model.mpc_forces, op2_model.load_vectors, #op2_model.grid_point_forces
                       ]
        fmt = '%sf' % (nnodes * 6)
        


        vtk_file.write('POINT_DATA %i\n' % nnodes)
        vtk_file.write('\n')
        vtk_file.write('\n')        
        for cases in nodal_cases:
            keys = list(cases.keys())
            if not keys:
                continue
            key0 = keys[0]
            node_ids = cases[key0].node_gridtype[:, 0]

            if nnodes == len(node_ids):
                # every node exists
                i = arange(nnodes)
                ni = nnodes
            else:
                # node_ids is a subset of nids
                i = searchsorted(nids, node_ids)
                ni = len(i)
                

            names = ['T1', 'T2', 'T3', 'R1', 'R2', 'R3']
            for isubcase, case in sorted(cases.items()):
                if case.is_real:

                    data = case.data[:, i, :]
                    ntimes = case.data.shape[0]
                    case_type = case.__class__.__name__
                    for itime in range(ntimes):
                        #if 0:
                            #for icol, name in enumerate(names):
                                #title = '%s_%s_isubcase=%s_itime=%s' % (case_type, name, isubcase, itime)
                                #vtk_file.write('SCALARS %s float\n' % title)
                                #vtk_file.write('LOOKUP_TABLE default\n')
                                ##datai = data[itime, i, icol]
                                #vtk_file.write(pack_float_1d_array(fmt, data[itime, i, icol]))
                        if 1:
                            title = '%s_isub=%s_itime=%s' % (case_type, isubcase, itime)


                           
                            #FIELD RealDisplacementArray_FIELD_isubcase=1_itime=0 6
                            #t1 1 72 float
                            #0.00764469 0.00762899 ...

                            title = title.replace(" ", "_")
                            title = title.replace(",", "_")
                            title = title.replace("=", "_")
                            title = title.replace("'", "_")
#                            title = title.replace("""""", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("Array", "_")
                            title = title.replace("__", "_")
                            
                            
#                            vtk_file.write('FIELD %s 6\n' % title)



                            vtk_file.write('\n')
                            for icol, name in enumerate(names):
                                
                                scalari =  'SCALARS ' + name + '_' + title + ' float'
                                vtk_file.write(scalari)
                                vtk_file.write('\n')


                                
                                tavola = 'LOOKUP_TABLE ' + name + '_' + title
                                vtk_file.write(tavola)

      

                                vtk_file.write('\n')

#                                vtk_file.write('%s 1 %s float\n' % (name, ni))
                                #datai = case.data[itime, i, icol]
                                vtk_file.write(pack_float_1d_array(fmt, data[itime, i, icol]))
                                vtk_file.write('\n')
                            #if 0:
                                #title = '%s_FIELD_isubcase=%s_itime=%s' % (case_type, isubcase, itime)
                                #vtk_file.write('FIELD %s 6 %i float\n' % (title, ni))
                                #vtk_file.write('LOOKUP_TABLE default\n')
                                #vtk_file.write(pack_float_2d_array(fmt, data[itime, i, :]))
                                
                                
                             
                                #print (icol)
                                
                                if icol == 0:
                                    
                                    vettori =  'VECTORS V_' + title + ' float'
                                    vtk_file.write(vettori)
                                    vtk_file.write('\n')


                                    vtk_file.write(pack_float_1d_array(fmt, data[itime, i, :3]))
                                    vtk_file.write('\n')



###----Modified by Marco. The original line is commented. Section start
#            vtk_file.write('MaterialID %i float\n' % nmaterials)

        vtk_file.write('CELL_DATA %i\n' % nelements)

        element_cases_1d = [op2_model.crod_stress, op2_model.crod_strain, op2_model.cbar_stress,
                            op2_model.cbar_strain, op2_model.cbeam_stress, op2_model.cbeam_strain]

        dict_extr_el_1d = {'RealRodStressArray' :  estraiel1,
                            'RealRodStrainArray' :  estraiel1,
                            'RealBarStressArray' :  estraiel1,
                            'RealBarStrainArray' :  estraiel1,
                            'RealBeamStressArray' :  estraiel2,
                            'RealBeamStrainArray' :  estraiel2}
                            

        element_cases_2d_PS = [op2_model.ctria3_stress, op2_model.cquad4_stress, op2_model.ctria3_strain,
                            op2_model.cquad4_strain]
        
        
        
        element_cases_2d_PC = [op2_model.cquad4_composite_stress, op2_model.ctria3_composite_stress,
                               op2_model.cquad4_composite_strain, op2_model.ctria3_composite_strain]
        
       
        
        
        
        dict_extr_el_2d = {'RealPlateStrainArray' :  estraiel2,
                            'RealPlateStressArray' :  estraiel2,
                            'RealCompositePlateStressArray' :  estraiel3,
                            'RealCompositePlateStrainArray' :  estraiel3}
        
        
      
        element_cases_3d_stress = [op2_model.ctetra_stress, op2_model.cpenta_stress, op2_model.chexa_stress]

        element_cases_3d_strain = [op2_model.ctetra_strain, op2_model.cpenta_strain, op2_model.chexa_strain]


        dict_extr_el_3d = {'RealSolidStressArray' :  estraiel2,
                            'RealSolidStrainArray' :  estraiel2}

#-----------------------1d----------------------------------------


        for cases in element_cases_1d:

#            #print ("eids 1d =", eids)


            keys = list(cases.keys())
            if not keys:
                continue
            key0 = keys[0]
            tipoarray = cases[key0].__class__.__name__
            
            ueids = dict_extr_el_1d[tipoarray](cases[key0])
            
            
            

            if bdf_nelements == len(ueids):
                # every element exists
                i = arange(bdf_nelements)
                ni = bdf_nelements
            else:
                # element_ids is a subset of eidszero
                i = searchsorted(eids, ueids)
                ni = len(i)
                i = arange(ni)

            names = cases[key0].get_headers()
            quanticasi = len(names)
            
            fmt = '%sf' % (bdf_nelements * quanticasi)
            
            for isubcase, case in sorted(cases.items()):
                if case.is_real:
                    #if i is None:
                        #data = case.data
                        #ni = nnodes
                    #else:
                        #data = zeros((nnodes, 6), dtype='float32')
                        #data[:, i, :] = case.data
                    data = case.data[:, i, :]
                    ntimes = case.data.shape[0]
                    case_type = case.__class__.__name__
                    for itime in range(ntimes):

                        if 1:
                            title = '%s_isub=%s_itime=%s' % (case_type, isubcase, itime)


                            title = title.replace(" ", "_")
                            title = title.replace(",", "_")
                            title = title.replace("=", "_")
                            title = title.replace("'", "_")
#                            title = title.replace("""""", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("Array", "_")
                            title = title.replace("__", "_")
                            

                            vtk_file.write('\n')
                            for icol, name in enumerate(names):
                                
                                scalari =  'SCALARS ' + name + '_' + title + ' float'
                                vtk_file.write(scalari)
                                vtk_file.write('\n')


                                
                                tavola = 'LOOKUP_TABLE ' + name + '_' + title
                                vtk_file.write(tavola)
                                vtk_file.write('\n')
                                listares = quasinumpyreplace(eidszero, i, data[itime, i, icol])

                                        
#                                ciccia = ' '.join([str(val) for val in listares]) + '\n'        
                                vtk_file.write(pack_float_1d_array(fmt, listares))


                                vtk_file.write('\n')


#-----------------------2d-PSHELL------------------------------------

        for cases in element_cases_2d_PS:


            keys = list(cases.keys())
            if not keys:
                continue
            
            key0 = keys[0]  

            tipoelemento = cases[key0].element_name
            
            if str(tipoelemento)[:5] == 'CQUAD':
                lungharray = 10
                
            else:
                lungharray = 2                


            tipoarray = cases[key0].__class__.__name__
            
            ueids = dict_extr_el_2d[tipoarray](cases[key0])
            
            
            
            if bdf_nelements == len(ueids):
                # every element exists
                i = arange(bdf_nelements)
                ni = bdf_nelements
            else:
                # element_ids is a subset of eidszero
                i = searchsorted(eids, ueids)

                ni = len(i)
                
                i = arange(lungharray * ni)
                
                

            names = cases[key0].get_headers()
            quanticasi = len(names)
            
           
            fmt = '%sf' % (bdf_nelements * quanticasi)
            
            for isubcase, case in sorted(cases.items()):
                if case.is_real:
                    #if i is None:
                        #data = case.data
                        #ni = nnodes
                    #else:
                        #data = zeros((nnodes, 6), dtype='float32')
                        #data[:, i, :] = case.data
                    data = case.data[:, i, :]
                    ntimes = case.data.shape[0]
                    case_type = case.__class__.__name__
                    for itime in range(ntimes):

                        if 1:
                            title = '%s_isube=%s_itime=%s' % (case_type, isubcase, itime)


                            title = title.replace(" ", "_")
                            title = title.replace(",", "_")
                            title = title.replace("=", "_")
                            title = title.replace("'", "_")
#                            title = title.replace("""""", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("Array", "_")
                            title = title.replace("__", "_")
                            

                            vtk_file.write('\n')
                            
                            semiarray = int(lungharray / 2)
                            
                            sotto = semiarray * arange(0, len(ueids) * 2, 2)
                            sopra = sotto + 1


                            vtk_file.write('\n')

                            
                            for icol, name in enumerate(names):
                                
                                scalari =  'SCALARS LW_' + tipoelemento + '_' + name + '_' + title + ' float'
                                vtk_file.write(scalari)
                                vtk_file.write('\n')


                                
                                tavola = 'LOOKUP_TABLE LW_' + tipoelemento + '_' + name + '_' + title
                                vtk_file.write(tavola)
                                vtk_file.write('\n')
                                        
                                listares = quasinumpyreplace(eidszero, searchsorted(eids, ueids), data[itime, sotto, icol])

                                vtk_file.write(pack_float_1d_array(fmt, listares))

                                vtk_file.write('\n')


                            vtk_file.write('\n')

                            
                            for icol, name in enumerate(names):
                                
                                scalari =  'SCALARS UP_' + tipoelemento + '_' + name + '_' + title + ' float'
                                vtk_file.write(scalari)
                                vtk_file.write('\n')


                                
                                tavola = 'LOOKUP_TABLE UP_' + tipoelemento + '_' + name + '_' + title
                                vtk_file.write(tavola)
                                vtk_file.write('\n')
                                        
                                listares = quasinumpyreplace(eidszero, searchsorted(eids, ueids), data[itime, sopra, icol])
                                vtk_file.write(pack_float_1d_array(fmt, listares))


                                vtk_file.write('\n')



#------------------------------2d-PCOMP----------------------------------------

        for cases in element_cases_2d_PC:
            

            
            keys = list(cases.keys())
            if not keys:
                continue
            
            key0 = keys[0]  

            tipoelemento = cases[key0].element_name
                    
            
            tipoarray = cases[key0].__class__.__name__

            ueids = dict_extr_el_2d[tipoarray](cases[key0])
            

            
            posizionielecomp = searchsorted(eids, ueids)
            
            
            
           
            
            
            
            
#----list of layers            
            
            arrayply = cases[key0].element_layer[:, 1]
            
            
            lenarrayply = len(arrayply)
            
            uno = where(arrayply == 1)
            uno = uno[0][1:]
            
            
#            arrayplaysplit = array_split(arrayply, uno)
            
            
            

#---max layer label


            maxlayer = max(cases[key0].element_layer[:, 1])
            
            
            
#-----


            listacompleta = array(list(arange(1, maxlayer + 1)) * len(ueids))
            
            

            uno = where(listacompleta == 1)
            uno = uno[0][1:]
            
            
#            listacompletasplit = array_split(listacompleta, uno)
            
            
            inserti = []
            
            for plai in arange(len(listacompleta)):
                if not arrayply[plai] == listacompleta[plai]:
                    inserti.append(plai)
                    arrayply = insert(arrayply, plai, 0)
            
            

#----
            
            keys = list(cases.keys())
            
            if not keys:
                continue
            key0 = keys[0]
            tipoarray = cases[key0].__class__.__name__
            
            
            if bdf_nelements == len(ueids):
                # every element exists
#                i = arange(bdf_nelements) 
                ni = bdf_nelements * len(arrayply)
            else:
                ni = lenarrayply
                
            i = arange(ni)

            eidszero = zeros(bdf_nelements)

                
                

            names = cases[key0].get_headers()
            quanticasi = len(names)
            
            fmt = '%sf' % (bdf_nelements * quanticasi * len(arrayply))
            
            for isubcase, case in sorted(cases.items()):
                if case.is_real:
                    data = case.data[:, i, :]
                    ntimes = case.data.shape[0]
                    case_type = case.__class__.__name__
                    for itime in range(ntimes):

                        vtk_file.write('\n')


                        if 1:
                            title = '%s_isube=%s_itime=%s' % (case_type, isubcase, itime)


                            title = title.replace(" ", "_")
                            title = title.replace(",", "_")
                            title = title.replace("=", "_")
                            title = title.replace("'", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("Array", "_")
                            title = title.replace("__", "_")
                            

                            vtk_file.write('\n')

                            for icol, name in enumerate(names):
                                    
    

#----temporary data matrix for PCOMP

                                
                                
                                
                                
                                risultatoply1 = data[itime, i, icol]
                                
                                
                                risultatoply = insert(risultatoply1, inserti, 0.)
                                
                                


                                for ply in arange(1, maxlayer + 1):
                                    
                                    resplayi = [risultatoply[(ply - 1) + jj * maxlayer]  for jj in arange(len(ueids))]
                                    
          
                                    scalari =  'SCALARS ply_' + (str(ply)) + '_' + tipoelemento + '_' + name + '_' + title + ' float'
                                    vtk_file.write(scalari)
                                    vtk_file.write('\n')
        
                                    
                                    tavola = 'LOOKUP_TABLE ply_' + (str(ply)) + '_' + tipoelemento + '_' + name + '_' + title
                                    vtk_file.write(tavola)
                                    vtk_file.write('\n')
        
                                    
                                    
                                    listares = quasinumpyreplace(eidszero, posizionielecomp, resplayi)
        
                                    vtk_file.write(pack_float_1d_array(fmt, listares))
        
                                    vtk_file.write('\n')



#-----------------------3d stress---------------------------------------



        dict_passi_3d = {'CTETRA' :  5,
                            'CPENTA' :  7,
                            'CHEXA' :  9}
        

#        passi = [len(bdf_model.Element(jj).nodes) for jj in elementi]
        



#        ezzerie = zeros(bdf_nelements, float) 
        casisolidi = ['oxx', 'oyy', 'ozz', 'txy', 'tyz', 'txz', 'omax', 'omid', 'omin', 'von_mises']

        for cases in element_cases_3d_stress:


            keys = list(cases.keys())
            if not keys:
                continue
            key0 = keys[0]
            ##print(key0)

            tipoarray = cases[key0].__class__.__name__
            
            tipoelemento = cases[key0].element_name
            
    
            
            ueids = dict_extr_el_3d[tipoarray](cases[key0])
            
            
            if bdf_nelements == len(ueids):
                # every element exists
                i = arange(bdf_nelements)
                ni = bdf_nelements
            else:
                # element_ids is a subset of eidszero
#                i = searchsorted(eids, ueids)
                i = searchsorted(elementi, ueids)
                #print("i1 = ", i)
            #print("i2 = ", i)
            #print(type(i))            

            names = cases[key0].get_headers()
            quanticasi = len(names)
            
            fmt = '%sf' % (bdf_nelements * quanticasi)
                                            



            for isubcase, case in sorted(cases.items()):
                if case.is_real:
                    #if i is None:
                        #data = case.data
                        #ni = nnodes
                    #else:
                        #data = zeros((nnodes, 6), dtype='float32')
                        #data[:, i, :] = case.data
                    #print ("tutti i dati    ",   case.data[:, :, :])
                    data = case.data[:, :, :]
                    
                    
                    ntimes = case.data.shape[0]
                    
                    #print("ntimes  ", ntimes)
                    
                    
                    
                    
                    passo = [dict_passi_3d[tipoelemento] for rrr in ueids]
                    
                    #print ("passo  ", passo)                    
                    passo1 = cumsum(passo)                    
                    #print ("passo1  ", passo1)                    
                    passo0 = passo1 - passo[0]                    
                    #print ("passo0  ", passo0)                   
                    
                    
                                
                                
                    case_type = case.__class__.__name__


                    for itime in range(ntimes):


                        if 1:
                            title = '%s_isub=%s_itime=%s' % (case_type, isubcase, itime)


                            title = title.replace(" ", "_")
                            title = title.replace(",", "_")
                            title = title.replace("=", "_")
                            title = title.replace("'", "_")
#                            title = title.replace("""""", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("Array", "_")
                            title = title.replace("__", "_")

#                            res_tank = []

                            vtk_file.write('\n')

#                            elementi1 = array(list(bdf_model.elements.keys()))
                           
                            
                            for icol, name in enumerate(names):
                                
                                #print("i, icol, name", i,"   ", icol, "    ", name)
                                listares0 = data[itime, passo0, icol]
                                
                                listares = quasinumpyreplace(eidszero, searchsorted(eids, ueids), listares0)
                                
                                        

                                title = 'Solid_Stress_' + '_' + str(name) + '_' + str(tipoelemento) + '_' + str(isubcase) + '_' + str(itime)
                                title = title.replace(" ", "_")
                                title = title.replace(",", "_")
                                title = title.replace("=", "_")
                                title = title.replace("'", "_")
                                title = title.replace("__", "_")
                                title = title.replace("__", "_")
                                title = title.replace("__", "_")
                                title = title.replace("Array", "_")
                                title = title.replace("__", "_")
                    
                                scalari =  'SCALARS ' + title + ' float'
                                vtk_file.write(scalari)
                                vtk_file.write('\n')
                                
                                tavola = 'LOOKUP_TABLE ' + title
                                vtk_file.write(tavola)
                                vtk_file.write('\n')
                                
                                #print("title",    title)
                                
                                #print("listares2  ", listares)
            
                                vtk_file.write(pack_float_1d_array(fmt, listares))

#---------------------------------------------3d strain--------------------------------------------------
        

#        passi = [len(bdf_model.Element(jj).nodes) for jj in elementi]
        
        #print("passi,   ", passi)



        for cases in element_cases_3d_strain:

            keys = list(cases.keys())
            #print("keys,    ", keys)
            if not keys:
                continue
            key0 = keys[0]
            #print("key0,    ", key0)

            tipoarray = cases[key0].__class__.__name__
            
            #print ("tipoarray", tipoarray)
            tipoelemento = cases[key0].element_name
            
            #print("tipoelemento 3d",    tipoelemento)   
    
            
            ueids = dict_extr_el_3d[tipoarray](cases[key0])
            
            
            if bdf_nelements == len(ueids):
                # every element exists
                i = arange(bdf_nelements)
                ni = bdf_nelements
            else:
                i = searchsorted(elementi, ueids)
                #print("i1 = ", i)
            #print("i2 = ", i)
            #print(type(i))            

            names = cases[key0].get_headers()
            quanticasi = len(names)
            
            fmt = '%sf' % (bdf_nelements * quanticasi)
                                            



            for isubcase, case in sorted(cases.items()):
                if case.is_real:
                    data = case.data[:, :, :]
                    
                    
                    ntimes = case.data.shape[0]
                    
                    
                    
                    passo = [dict_passi_3d[tipoelemento] for rrr in ueids]
                    
                    passo1 = cumsum(passo)                    
                    passo0 = passo1 - passo[0]                    
                    
                                
                    case_type = case.__class__.__name__


                    for itime in range(ntimes):


                        if 1:
                            title = '%s_isub=%s_itime=%s' % (case_type, isubcase, itime)


                            title = title.replace(" ", "_")
                            title = title.replace(",", "_")
                            title = title.replace("=", "_")
                            title = title.replace("'", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("__", "_")
                            title = title.replace("Array", "_")
                            title = title.replace("__", "_")

#                            res_tank = []

                            vtk_file.write('\n')

#                            elementi1 = array(list(bdf_model.elements.keys()))
                           
                            
                            for icol, name in enumerate(names):
                                
                                #print("i, icol, name", i,"   ", icol, "    ", name)
                                
                                listares0 = data[itime, passo0, icol]
                                
                                listares = quasinumpyreplace(eidszero, searchsorted(eids, ueids), listares0)
                                
                                title = 'Solid_Strain_' + '_' + str(name) + '_' + str(tipoelemento) + '_' + str(isubcase) + '_' + str(itime)
                                title = title.replace(" ", "_")
                                title = title.replace(",", "_")
                                title = title.replace("=", "_")
                                title = title.replace("'", "_")
                                title = title.replace("__", "_")
                                title = title.replace("__", "_")
                                title = title.replace("__", "_")
                                title = title.replace("Array", "_")
                                title = title.replace("__", "_")
                    
                                scalari =  'SCALARS ' + title + ' float'
                                vtk_file.write(scalari)
                                vtk_file.write('\n')
                                
                                tavola = 'LOOKUP_TABLE ' + title
                                vtk_file.write(tavola)
                                vtk_file.write('\n')
                                
                                #print("title",    title)
                                
                                #print("listares2  ", listares)
            
                                vtk_file.write(pack_float_1d_array(fmt, listares))



###----Modified by Marco. Section end




        #CELLS 217 1039
    return bdf_model, op2_model

def main(model):
    export_to_vtk(model)

if __name__ == '__main__':
    model = ''    
    try:
        model = sys.argv[1]
    except Exception as e:
        print('\nInsert the model name.\n')
        raise e
    main(model)
