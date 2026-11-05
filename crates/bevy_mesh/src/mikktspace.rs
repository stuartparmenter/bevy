use crate::{GenerateTangentsError, Indices};

struct MikktspaceGeometryHelper<'a> {
    indices: Option<&'a Indices>,
    positions: &'a Vec<[f32; 3]>,
    normals: &'a Vec<[f32; 3]>,
    uvs: &'a Vec<[f32; 2]>,
    tangents: Vec<[f32; 4]>,
}

impl MikktspaceGeometryHelper<'_> {
    fn index(&self, face: usize, vert: usize) -> usize {
        let index_index = face * 3 + vert;

        match self.indices {
            Some(Indices::U16(indices)) => indices[index_index] as usize,
            Some(Indices::U32(indices)) => indices[index_index] as usize,
            None => index_index,
        }
    }
}

impl bevy_mikktspace::Geometry for MikktspaceGeometryHelper<'_> {
    fn num_faces(&self) -> usize {
        self.indices
            .map(Indices::len)
            .unwrap_or_else(|| self.positions.len())
            / 3
    }

    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.index(face, vert)]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.index(face, vert)]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.index(face, vert)]
    }

    fn set_tangent(
        &mut self,
        tangent_space: Option<bevy_mikktspace::TangentSpace>,
        face: usize,
        vert: usize,
    ) {
        self.tangents[face * 3 + vert] = tangent_space.unwrap_or_default().tangent_encoded();
    }
}

pub(crate) fn generate_tangents(
    indices: Option<&Indices>,
    positions: &Vec<[f32; 3]>,
    normals: &Vec<[f32; 3]>,
    uvs: &Vec<[f32; 2]>,
) -> Result<Vec<[f32; 4]>, GenerateTangentsError> {
    let mut mikktspace_mesh = MikktspaceGeometryHelper {
        indices,
        positions,
        normals,
        uvs,
        tangents: vec![[0.; 4]; indices.map_or(positions.len(), Indices::len) / 3 * 3],
    };
    bevy_mikktspace::generate_tangents(&mut mikktspace_mesh)?;

    Ok(mikktspace_mesh.tangents)
}
